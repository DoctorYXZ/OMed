#1 ViT extracts the features of 224x224 pixel patches
import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
from transformers import ViTFeatureExtractor, ViTModel

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

# Load the pretrained Vision Transformer model
feature_extractor = ViTFeatureExtractor.from_pretrained('C:\\code\\Prognosis_prediction\\vit_large_patch16_224')
model = ViTModel.from_pretrained('C:\\code\\Prognosis_prediction\\vit_large_patch16_224')
model.eval()
if use_cuda and torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs.")
    model = torch.nn.DataParallel(model)
model.to(device)

# Define the folder path for color-normalized 224x224 patches files
folder_path = 'C:\\path\\to\\your color-normalized 224patch files'
features_dict = {}

# Iterate through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.tif'):
        image_path = os.path.join(folder_path, filename)
        with Image.open(image_path) as image:
            wsi_image = np.array(image)[:, :, :3]
            wsi_image_pil = Image.fromarray(wsi_image)
            wsi_image_tensor = feature_extractor(images=wsi_image_pil, return_tensors="pt")["pixel_values"].to(device)
            with torch.no_grad():
                outputs = model(wsi_image_tensor)
                features = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
                features_dict[filename] = features

# Save features to CSV
output_csv_path = 'C:\\path\\to\\1features.csv'
features_df = pd.DataFrame.from_dict(features_dict, orient='index')
features_df.to_csv(output_csv_path)





#2 Compute statistics of the features and save to a new CSV
file_path = output_csv_path
df = pd.read_csv(file_path)
stats_df = pd.DataFrame()

for i in range(1, 2500):
    file_pattern = f'RETRO_HE{i}.tif'
    df_current = df[df['Unnamed: 0'].str.contains(file_pattern)]
    if not df_current.empty:
        stats = df_current.iloc[:, 1:].describe()
        stats_df[f'RETRO_HE{i}_Mean'] = stats.loc['mean']
        stats_df[f'RETRO_HE{i}_Median'] = stats.loc['50%']
        stats_df[f'RETRO_HE{i}_Std'] = stats.loc['std']
        stats_df[f'RETRO_HE{i}_25%'] = stats.loc['25%']
        stats_df[f'RETRO_HE{i}_75%'] = stats.loc['75%']

stats_file_path = 'C:\\path\\to\\2stats_features.csv'
stats_df.to_csv(stats_file_path)





#3 Adjust data structure for further analysis
stats_df = pd.read_csv(stats_file_path, index_col=0)
relevant_columns = [col for col in stats_df.columns if 'Unnamed' not in col]
new_column_labels = [f'RETRO_HE{i}' for i in range(1, 2500)]
new_row_labels = []
for i in range(1024):
    for stat in ['Mean', 'Median', 'Std', '25%', '75%']:
        new_row_labels.append(f'{i}_{stat}')

adjusted_df = pd.DataFrame(index=new_row_labels, columns=new_column_labels)
for col in relevant_columns:
    file_number, stat_type = col.split('_')[1], col.split('_')[2]
    for i in range(1024):
        new_row_label = f'{i}_{stat_type}'
        new_column_label = f'RETRO_HE{file_number}'
        adjusted_df.at[new_row_label, new_column_label] = stats_df[col].iloc[i]

adjusted_df.dropna(axis=1, how='all', inplace=True)
final_adjusted_file_path = 'C:\\path\\to\\3adjusted_stats_features.csv'
adjusted_df.to_csv(final_adjusted_file_path)






#4 Transpose the DataFrame and save
df = pd.read_csv(final_adjusted_file_path)
df_reset = df.reset_index()
transposed_df = df_reset.T
new_header = transposed_df.iloc[0]
transposed_df = transposed_df[1:]
transposed_df.columns = new_header
output_file_path = 'C:\\path\\to\\4retro_features.csv'   #The features of the retrospective training set, external and prospective testing sets were extracted, and combined into retro_external_pro_features.csv
transposed_df.to_csv(output_file_path, index_label='Original_Columns')

print(f"All processes completed. Final data saved to {output_file_path}")
