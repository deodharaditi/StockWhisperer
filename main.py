import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_absolute_percentage_error
import json

#work_dir= r"D:\\Northeastern\\Spring 2024\\ADSA\\capstone\\" test_data.json


#Model Implementation

class CNN_BiLSTM_AM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, sequence_length, dropout_prob):
        super(CNN_BiLSTM_AM, self).__init__()
        self.conv1d_1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1, padding=0, bias = True)
        self.conv1d_2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=0, bias = True)
        self.conv1d_3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, padding=0, bias = True)
        self.pool = nn.MaxPool1d(kernel_size=1, padding=0)
        self.lstm_1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=True, bias = True)
        self.lstm_2 = nn.LSTM(hidden_dim * 2, hidden_dim, batch_first=True, bidirectional=True, bias = True)
        self.attention_linear = nn.Linear(hidden_dim * 2, 1, bias = True)
        self.linear = nn.Linear(hidden_dim * 2, output_dim, bias = True)
        self.sequence_length = sequence_length
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, input_dim)
        batch_size, sequence_length, input_dim = x.size()

        # Reshape for Conv1d input
        x = x.permute(0, 2, 1)  # Reshape for Conv1d input
        x = self.pool(F.relu(self.conv1d_1(x)))
        x = self.pool(F.relu(self.conv1d_2(x)))
        x = self.pool(F.relu(self.conv1d_3(x)))

        x = self.dropout(x)
        # Reshape for LSTM input
        x = x.permute(0, 2, 1)  # Reshape for LSTM input
        x, _ = self.lstm_1(x)
        x, _ = self.lstm_2(x)
        x = self.dropout(x)
        # Attention mechanism
        attention_weights = F.softmax(self.attention_linear(x), dim=1)
        attended_out = torch.sum(attention_weights * x, dim=1)

        # Output layer
        output = self.linear(attended_out)

        return output


def load_data(work_dir):
  with open(f"test_data.json", 'r') as file:
    test_data = json.load(file)
 
  with open(f"mean.json", 'r') as file:
    mean = json.load(file)
  
  with open(f"std.json", 'r') as file:
    std = json.load(file)

  return test_data, mean, std

def parameter_updater(data,stock,parameters):
  columns = ['revenue',
 'pretax_income',
 'income_tax',
 'net_income_continuing',
 'net_income_discontinued',
 'net_income',
 'preferred_dividends',
 'net_income_available_to_shareholders',
 'eps_basic',
 'eps_diluted',
 'shares_basic',
 'shares_diluted',
 'cash_and_equiv',
 'ppe_net',
 'intangible_assets',
 'goodwill',
 'other_lt_assets',
 'total_assets',
 'st_debt',
 'lt_debt',
 'other_lt_liabilities',
 'total_liabilities',
 'common_stock',
 'preferred_stock',
 'retained_earnings',
 'aoci',
 'apic',
 'treasury_stock',
 'other_equity',
 'total_equity',
 'total_liabilities_and_equity',
 'cfo_net_income',
 'cfo_da',
 'cfo_receivables',
 'cfo_inventory',
 'cfo_prepaid_expenses',
 'cfo_other_working_capital',
 'cfo_change_in_working_capital',
 'cfo_deferred_tax',
 'cfo_stock_comp',
 'cfo_other_noncash_items',
 'cf_cfo',
 'cfi_ppe_net',
 'cfi_acquisitions_net',
 'cfi_investment_net',
 'cfi_intangibles_net',
 'cfi_other',
 'cf_cfi',
 'cff_common_stock_net',
 'cff_pfd_net',
 'cff_debt_net',
 'cff_dividend_paid',
 'cff_other',
 'cf_cff',
 'cf_forex',
 'cf_net_change_in_cash',
 'capex',
 'ebitda',
 'book_value',
 'tangible_book_value',
 'roa',
 'roe',
 'roic',
 'roce',
 'ebitda_margin',
 'pretax_margin',
 'net_income_margin',
 'assets_to_equity',
 'equity_to_assets',
 'debt_to_equity',
 'revenue_per_share',
 'ebitda_per_share',
 'pretax_income_per_share',
 'book_value_per_share',
 'tangible_book_per_share',
 'market_cap',
 'enterprise_value',
 'price_to_earnings',
 'price_to_book',
 'price_to_tangible_book',
 'price_to_sales',
 'enterprise_value_to_earnings',
 'enterprise_value_to_book',
 'enterprise_value_to_tangible_book',
 'enterprise_value_to_sales',
 'revenue_growth',
 'ebitda_growth',
 'pretax_income_growth',
 'net_income_growth',
 'eps_diluted_growth',
 'shares_diluted_growth',
 'cash_and_equiv_growth',
 'total_assets_growth',
 'total_equity_growth',
 'revenue_cagr_10',
 'eps_diluted_cagr_10',
 'total_assets_cagr_10',
 'total_equity_cagr_10',
 'payout_ratio',
 'shares_eop',
 'dividends',
 'period_end_price',
 'pretax_margin_median',
 'roa_median',
 'roe_median',
 'roic_median',
 'assets_to_equity_median',
 'debt_to_equity_median',
 'Average_Stock_Value_After_Result']
  tmp_df = pd.DataFrame(data[stock], columns=columns)
  original_df = pd.DataFrame(data[stock], columns=columns)

  max_pct_change = 0
  for key,value in parameters.items():
    tmp_df.iloc[-1][key]=value
    pct_change = ((tmp_df.iloc[-1][key]-original_df.iloc[-1][key])/original_df.iloc[-1][key])*100
    if(abs(pct_change) > abs(max_pct_change)):
       max_pct_change = pct_change
  print(max_pct_change)
  for value in columns:
     original_df[value]=(original_df[value]*(100+max_pct_change)/100)
  return original_df.to_numpy()

def data_loader(data, mean, std, sequence_length, overlap):
  mean_vector = mean
  std_vector = std

  # Check for zero standard deviation
  zero_std_indices = np.where(std_vector == 0)[0]

  # Avoid division by zero and handle NaN
  std_vector[std_vector == 0] = 1  # Replace zero standard deviations with 1 to avoid division by zero

  # Normalize the data
  normalized_data = (data - mean_vector) / std_vector

  # Handle elements where standard deviation was zero
  normalized_data[:, zero_std_indices] = 0  # Set corresponding elements to zero

  # Assign the normalized data back to grouped_data
  data = normalized_data

  # using sliding window mechanism to split train, test and validation data
  dataset=[]
  for j in range(len(data)):#5
      if j*(sequence_length - overlap)+sequence_length < len(data):
          datapoint = data[j*(sequence_length - overlap):j*(sequence_length - overlap)+sequence_length]
          dataset.append(datapoint)
      else:
          break

  dataset = np.array(dataset)
  features = dataset[:, :, :-1]

  targets = []
  for j in range(len(data)):#5
      if j*(sequence_length - overlap)+sequence_length < len(data):
          target = data[j*(sequence_length - overlap)+sequence_length][-1]
          targets.append(target)
      else:
          break

  targets = np.array(targets)
  features = torch.tensor(features, dtype=torch.float32)
  targets = torch.tensor(targets, dtype=torch.float32)

  test_dataset = TensorDataset(features, targets)
  test_loader = DataLoader(test_dataset, shuffle=False)

  return test_loader


def load_model(model_name, input_dim, hidden_dim, output_dim, sequence_length, dropout_prob):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model = CNN_BiLSTM_AM(input_dim, hidden_dim, output_dim, sequence_length, dropout_prob).to(device)
  model.load_state_dict(torch.load(f"{model_name}", map_location=device))
  return model, device

def evaluate(model,test_loader, device):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=0.0001)
  train_losses = []
  model.eval()
  running_test_loss = 0.0
  predictions_test_data = []
  true_values_test_data = []
  with torch.no_grad():
    #    val_progress_bar = tqdm(val_dataloader, desc=f'Validation', leave=False, mininterval=1)
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)#labels are normalized

            # Forward pass
            normalized_outputs = model(inputs.float())
            loss = criterion(normalized_outputs.squeeze(), labels.float()) # losses are computed between normalized outputs and nromalized prediction

            running_test_loss += loss.item()
            true_values_test_data.extend(normalized_outputs.cpu().numpy())
            predictions_test_data.extend(labels.cpu().numpy())

        print(f"Test Loss for:", running_test_loss / len(test_loader))
  return predictions_test_data,true_values_test_data[0]

def denormalize(value,mean,std):
  return (value*std+mean)


def predict(input_param, input_stock):
  sequence_length = 4
  overlap = 3
  batch_size = 25
  input_dim = 108
  hidden_dim = 32
  output_dim = 1
  dropout_prob = 0.2
  parameters= input_param 
  stock = input_stock
  print(f"Stock Name: {stock}")

  t,m,s = load_data(work_dir)
  model, device = load_model('model.state', input_dim, hidden_dim, output_dim, sequence_length, dropout_prob)
  test_data = parameter_updater(t,stock,parameters)
  #print(test_data)
  test_loader = data_loader(test_data, np.array(m[stock]), np.array(s[stock]), sequence_length, overlap)

  pred, true = evaluate(model, test_loader, device)
  print(denormalize(pred[0], m[stock][-1],s[stock][-1]))
  print(denormalize(true[0], m[stock][-1],s[stock][-1]))

  return denormalize(pred[0], m[stock][-1],s[stock][-1])

if __name__ == '__main__':
    #main()
    print("Yes it worked.")
