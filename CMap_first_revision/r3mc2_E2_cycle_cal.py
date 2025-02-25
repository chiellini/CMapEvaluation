# import os
# import glob
# import pandas as pd
# import numpy as np

# cd_file_root=r'C:\Users\zelinli6\OneDrive - City University of Hong Kong - Student\Temporary Sharing\CDFiles'
#
# cd_files_this=glob.glob(os.path.join(cd_file_root,'*wee1*.csv'))
#
# time_list=[]
#
# for cd_file_path in cd_files_this:
#     pd_this=pd.read_csv(cd_file_path)
#     pd_E2_this=pd_this.loc[(pd_this['cell']=='Ea') | (pd_this['cell']=='Ep')]
#     start_time=min(pd_E2_this['time'])
#     end_time=max(pd_E2_this['time'])
#     time_list.append(end_time-start_time)
#
# print(np.mean(time_list)*1.42,np.std(time_list)*1.42)


from mnemonic import Mnemonic

# Your hex private key (64-character hex string)
hex_private_key = "33fd60c990e751c73bf7b97b832529572db6e58caf3d6193c10aa58461b9e6ad"

# Initialize the BIP-39 mnemonic generator
mnemo = Mnemonic("english")

# Convert hex private key to 12-word mnemonic
mnemonic_phrase = mnemo.to_mnemonic(bytes.fromhex(hex_private_key))

print("Mnemonic Phrase:", mnemonic_phrase)
