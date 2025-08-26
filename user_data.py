# user_data.py

import csv
import os

def save_user_info(name, phone, email, filename="user_info.csv"):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Name', 'Phone', 'Email'])
        if not file_exists:
            writer.writeheader()
        writer.writerow({'Name': name, 'Phone': phone, 'Email': email})
