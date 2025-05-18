import os
import re

root = 'MetaDriver/models'
models = ['PFENet', 'BAM', 'HDMNet', 'AMNet', 'AENet', 'MetaDriver']
datasets = ['metadada', 'metapsad']


new_dada_data_root = '/data/dataset/DADA'      # Replace it with your MetaDADA path.
new_psad_data_root = '/data/dataset/MetaPSAD'  # Replace it with your MetaPSAD path.

for model in models:
    for dataset in datasets:
        config_dir = os.path.join(root, model, f'config/{dataset}')
        if not os.path.isdir(config_dir):
            print(f"[skip] no exist ---> {config_dir}")
            continue

        for file in os.listdir(config_dir):
            if not file.endswith('.yaml'):
                continue

            file_path = os.path.join(config_dir, file)
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()

                new_lines = []
                replaced = False
                for line in lines:
                    # match data_root or base_data_root
                    if dataset == 'metadada':
                        modify_value = new_dada_data_root
                    else:
                        modify_value = new_psad_data_root

                    if re.match(r'^\s*data_root\s*:', line):
                        newline = re.sub(r':\s*.*$', f': {modify_value}', line)
                        new_lines.append(newline)
                        replaced = True
                    elif re.match(r'^\s*base_data_root\s*:', line):
                        newline = re.sub(r':\s*.*$', f': {modify_value}', line)
                        new_lines.append(newline)
                        replaced = True
                    else:
                        new_lines.append(line)

                if replaced:
                    with open(file_path, 'w') as f:
                        f.writelines(new_lines)
                    print(f"successful ---> {file_path}")
                else:
                    print(f"no exist ---> {file_path}")

            except Exception as e:
                print(f"error ---> {file_path}: {e}")