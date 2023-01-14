import json
import os
import torch
import torchaudio



root_dir = '/rds/user/jdb206/hpc-work/MLMI2/exp'
file = 'test.json'

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
    fbank = torchaudio.compliance.kaldi.fbank(waveform) #default sample_freq is 16khz, same as TIMIT
    
    _ , entry = key.split('_')
    
    entry = entry.split('.')[0] #Removes .WAV

    if not os.path.exists(spk_id_dir):
        os.mkdir(spk_id_dir)
    
    entry_path = os.path.join(spk_id_dir, entry)
    if not os.path.exists(entry_path):
        os.mkdir(entry_path)
    
    fbank_save_path = os.path.join(entry_path, 'fbank.pt')
    torch.save(fbank, fbank_save_path)

    #Make the dictionary using same key as before

    fbank_dict = {}
    fbank_dict["fbank"] = fbank_save_path
    fbank_dict["duration"] = dur
    fbank_dict["spk_id"] = spk_id
    fbank_dict["phn"] = phn

    output_json_dict[key] = fbank_dict
    # print(value["phn"])
    

output_json = os.path.join(root_dir, 'test_fbank.json')
with open(output_json, 'w') as json_file:
    json.dump(output_json_dict, json_file, indent=4)
