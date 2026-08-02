import pandas as pd
import os

nirf_data = [
    # Top IITs
    {"Institute Name": "Indian Institute of Technology Madras", "NIRF Rank": 1},
    {"Institute Name": "Indian Institute of Technology Delhi", "NIRF Rank": 2},
    {"Institute Name": "Indian Institute of Technology Bombay", "NIRF Rank": 3},
    {"Institute Name": "Indian Institute of Technology Kanpur", "NIRF Rank": 4},
    {"Institute Name": "Indian Institute of Technology Kharagpur", "NIRF Rank": 5},
    {"Institute Name": "Indian Institute of Technology Roorkee", "NIRF Rank": 6},
    {"Institute Name": "Indian Institute of Technology Guwahati", "NIRF Rank": 7},
    {"Institute Name": "Indian Institute of Technology Hyderabad", "NIRF Rank": 8},
    {"Institute Name": "National Institute of Technology Tiruchirappalli", "NIRF Rank": 9},
    {"Institute Name": "Indian Institute of Technology (BHU) Varanasi", "NIRF Rank": 10},
    {"Institute Name": "Indian Institute of Technology Indore", "NIRF Rank": 12},
    {"Institute Name": "National Institute of Technology Karnataka, Surathkal", "NIRF Rank": 13},
    {"Institute Name": "National Institute of Technology Rourkela", "NIRF Rank": 14},
    {"Institute Name": "Indian Institute of Technology (ISM) Dhanbad", "NIRF Rank": 15},
    {"Institute Name": "Indian Institute of Technology Gandhinagar", "NIRF Rank": 18},
    {"Institute Name": "Indian Institute of Technology Ropar", "NIRF Rank": 22},
    {"Institute Name": "National Institute of Technology Warangal", "NIRF Rank": 21},
    {"Institute Name": "Indian Institute of Technology Jodhpur", "NIRF Rank": 28},
    {"Institute Name": "Indian Institute of Technology Mandi", "NIRF Rank": 31},
    {"Institute Name": "National Institute of Technology Calicut", "NIRF Rank": 23},
    {"Institute Name": "Indian Institute of Technology Patna", "NIRF Rank": 34},
    {"Institute Name": "National Institute of Technology Durgapur", "NIRF Rank": 43},
    {"Institute Name": "National Institute of Technology Silchar", "NIRF Rank": 40},
    {"Institute Name": "Visvesvaraya National Institute of Technology, Nagpur", "NIRF Rank": 41},
    {"Institute Name": "National Institute of Technology Kurukshetra", "NIRF Rank": 58},
    {"Institute Name": "Motilal Nehru National Institute of Technology Allahabad", "NIRF Rank": 49},
    {"Institute Name": "Malaviya National Institute of Technology Jaipur", "NIRF Rank": 37},
    {"Institute Name": "Sardar Vallabhbhai National Institute of Technology, Surat", "NIRF Rank": 65},
    {"Institute Name": "Dr. B R Ambedkar National Institute of Technology, Jalandhar", "NIRF Rank": 46},
    {"Institute Name": "National Institute of Technology Meghalaya", "NIRF Rank": 72},
    {"Institute Name": "Maulana Azad National Institute of Technology Bhopal", "NIRF Rank": 80},
    {"Institute Name": "National Institute of Technology Raipur", "NIRF Rank": 70},
    {"Institute Name": "National Institute of Technology Agartala", "NIRF Rank": 91},
    {"Institute Name": "National Institute of Technology Goa", "NIRF Rank": 90},
    {"Institute Name": "National Institute of Technology Jamshedpur", "NIRF Rank": 85},
    {"Institute Name": "Indian Institute of Information Technology Allahabad", "NIRF Rank": 89},
    {"Institute Name": "National Institute of Technology Patna", "NIRF Rank": 56},
    {"Institute Name": "National Institute of Technology Hamirpur", "NIRF Rank": 101},
    {"Institute Name": "National Institute of Technology Puducherry", "NIRF Rank": 102},
    {"Institute Name": "National Institute of Technology Manipur", "NIRF Rank": 95},
    {"Institute Name": "National Institute of Technology Arunachal Pradesh", "NIRF Rank": 142},
    {"Institute Name": "National Institute of Technology Srinagar", "NIRF Rank": 105},
    {"Institute Name": "National Institute of Technology Delhi", "NIRF Rank": 51},
    {"Institute Name": "National Institute of Technology Mizoram", "NIRF Rank": 140},
    {"Institute Name": "National Institute of Technology Nagaland", "NIRF Rank": 150},
    {"Institute Name": "National Institute of Technology Sikkim", "NIRF Rank": 151},
    {"Institute Name": "National Institute of Technology Uttarakhand", "NIRF Rank": 152},
    {"Institute Name": "National Institute of Technology, Andhra Pradesh", "NIRF Rank": 153},
    
    # IIITs
    {"Institute Name": "Atal Bihari Vajpayee Indian Institute of Information Technology & Management Gwalior", "NIRF Rank": 103},
    {"Institute Name": "Indian Institute of Information Technology, Design & Manufacturing, Kancheepuram", "NIRF Rank": 184},
    {"Institute Name": "Pandit Dwarka Prasad Mishra Indian Institute of Information Technology, Design and Manufacturing (IIITDM) Jabalpur", "NIRF Rank": 97},
    {"Institute Name": "Indian Institute of Information Technology Guwahati", "NIRF Rank": 112},
    
    # New IITs
    {"Institute Name": "Indian Institute of Technology Bhubaneswar", "NIRF Rank": 47},
    {"Institute Name": "Indian Institute of Technology Tirupati", "NIRF Rank": 59},
    {"Institute Name": "Indian Institute of Technology Palakkad", "NIRF Rank": 69},
    {"Institute Name": "Indian Institute of Technology Jammu", "NIRF Rank": 67},
    {"Institute Name": "Indian Institute of Technology Dharwad", "NIRF Rank": 93},
    {"Institute Name": "Indian Institute of Technology Bhilai", "NIRF Rank": 81},
    {"Institute Name": "Indian Institute of Technology Goa", "NIRF Rank": 82}
]

df = pd.DataFrame(nirf_data)
output_path = r"d:\jeerankpred\data\nirf_2024.csv"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)
print(f"Created {output_path} with {len(df)} rows.")
