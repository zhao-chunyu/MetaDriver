import pandas as pd
import os

root = 'MetaDriver/models'
models = ['PFENet', 'BAM', 'HDMNet', 'AMNet', 'AENet', 'MetaDriver']
splits = [0, 1, 2, 3]
backbones = ['resnet50', 'vgg']

dataset = 'psad'


backbone_sheets = {}

for backbone in backbones:
    data = {}
    for model in models:
        row = []
        for split in splits:
            xlsx_path = os.path.join(root, model, f'exp/meta{dataset}/split{split}/{backbone}/model/metrics.xlsx')
        try:
            df = pd.read_excel(xlsx_path, sheet_name='Novel')
            last_row = df.tail(1).values.flatten().tolist()

            row += last_row[1:4]
        except Exception as e:
            print(f"Error reading {xlsx_path}: {e}")
            row += [None, None, None]
        data[model] = row

    columns = ['Methods']
    for split in splits:
        columns += [f'CC-{split}', f'SIM-{split}', f'KLD-{split}']

    df_out = pd.DataFrame([[method] + data[method] for method in models], columns=columns)
    backbone_sheets[backbone] = df_out


output_path = f'{dataset}_all_metrics.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for backbone, df in backbone_sheets.items():
        df.to_excel(writer, sheet_name=backbone, index=False)

print(f"over, saving ---> {output_path}")
