import os
import urllib.request
import sqlite3

def download_josaa_db():
    url = "https://github.com/Pardhavmaradani/josaa-sql-interface/raw/main/db/josaa-2016-2024-all.db"
    dest_dir = r"d:\jeerankpred\data\raw"
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "josaa-2016-2024-all.db")
    
    print(f"Downloading JoSAA database from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded to {dest_path}")
        
        # Verify file is valid SQLite and list tables
        conn = sqlite3.connect(dest_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        print("Tables in database:", [t[0] for t in tables])
        
        # Inspect the schema of the main table
        for table in tables:
            t_name = table[0]
            cursor.execute(f"PRAGMA table_info({t_name});")
            schema = cursor.fetchall()
            print(f"\nSchema for table '{t_name}':")
            for col in schema:
                print(f"  {col[1]} ({col[2]})")
            
            # Print row count
            cursor.execute(f"SELECT COUNT(*) FROM {t_name};")
            count = cursor.fetchone()[0]
            print(f"  Total rows: {count}")
            
            # Print sample row
            cursor.execute(f"SELECT * FROM {t_name} LIMIT 1;")
            sample = cursor.fetchone()
            print(f"  Sample row: {sample}")
            
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    download_josaa_db()
