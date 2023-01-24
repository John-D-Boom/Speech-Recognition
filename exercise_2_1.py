import json
import os
import torch
import torchaudio



root_dir = '/rds/user/jdb206/hpc-work/MLMI2/exp'
file = 'train.json'

# open the file and read in the contents
with open(os.path.join(root_dir, file)) as json_file:
    data = json.load(json_file)
    
# print out the data
output_json_dict = {}
for key, value in data.items():
    wav_path = value["wav"]
    spk_id = value["spk_id"]
    dur = value["duration"]
    phn = value["phn"]

    fbank_dir = os.path.join(root_dir, 'fbanks')
    spk_id_dir = os.path.join(fbank_dir, spk_id)

    #Load waveform, generate fbank, and save to fbanks/[spk_id]/fbank.pt
    waveform, sample_rate = torchaudio.load(wav_path)

    # speedup_transform = torchaudio.prototype.transforms.speed(sample_rate, 1.1)
    # slowdown_transform = torchaudio.prototype.transforms.speed(sample_rate, 0.9)
    # fast_wav = speedup_transform(waveform, len(waveform))
    # slow_wav = slowdown_transform(waveform, len(waveform))

    slow_effects = [
        ["speed", "0.9"],   # reduce the speed
                            # This only changes sample rate, so it is necessary to
                            # add `rate` effect with original sample rate after this.
        ["rate", f"{sample_rate}"],
    ]
    fast_effects = [
        ["speed", "1.1"],   # increase the speed
                            # This only changes sample rate, so it is necessary to
                            # add `rate` effect with original sample rate after this.
        ["rate", f"{sample_rate}"],
    ]

    # Apply effects
    waveform_slow, sample_rate_slow = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, slow_effects)
    
    waveform_fast, sample_rate_fast = torchaudio.sox_effects.apply_effects_tensor(
        waveform, sample_rate, fast_effects)

    fbank_slow = torchaudio.compliance.kaldi.fbank(waveform_slow, sample_frequency= sample_rate_slow)
    fbank_fast = torchaudio.compliance.kaldi.fbank(waveform_fast, sample_frequency= sample_rate_fast)

    # fbank = torchaudio.compliance.kaldi.fbank(waveform) #default sample_freq is 16khz, same as TIMIT
    
    _ , entry = key.split('_')
    
    entry = entry.split('.')[0] #Removes .WAV

    if not os.path.exists(spk_id_dir):
        os.mkdir(spk_id_dir)
    
    entry_path = os.path.join(spk_id_dir, entry)
    if not os.path.exists(entry_path):
        os.mkdir(entry_path)
    
    fbank_save_path = os.path.join(entry_path, 'fbank.pt')
    slow_fbank_save_path = os.path.join(entry_path, 'fbank_0_9.pt')
    fast_fbank_save_path = os.path.join(entry_path, 'fbank_1_1.pt')
    torch.save(fbank_slow, slow_fbank_save_path)
    torch.save(fbank_fast, fast_fbank_save_path)


    #Make the dictionary using same key as before

    fbank_dict = {}
    fbank_dict["fbank"] = fbank_save_path
    fbank_dict["duration"] = dur
    fbank_dict["spk_id"] = spk_id
    fbank_dict["phn"] = phn
    fbank_dict["fbank_0_9"] = slow_fbank_save_path
    fbank_dict["fbank_1_1"] = fast_fbank_save_path


    output_json_dict[key] = fbank_dict
    # print(value["phn"])
    

output_json = os.path.join(root_dir, 'train_fbank_speeds.json')
with open(output_json, 'w') as json_file:
    json.dump(output_json_dict, json_file, indent=4)
