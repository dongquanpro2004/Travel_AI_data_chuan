import pandas as pd

def generate_database():
    data = [
        {'Destination': 'Đà Lạt', 'Name': 'Khu du lịch Langbiang', 'Category': 'Khám phá', 'Cost': 150000},
        {'Destination': 'Đà Lạt', 'Name': 'Chợ đêm Đà Lạt', 'Category': 'Ẩm thực', 'Cost': 200000},
        {'Destination': 'Đà Lạt', 'Name': 'Quán Cafe Túi Mơ To', 'Category': 'Thư giãn', 'Cost': 80000},
        {'Destination': 'Đà Lạt', 'Name': 'Thác Datanla', 'Category': 'Khám phá', 'Cost': 250000},
        {'Destination': 'Vũng Tàu', 'Name': 'Tượng Chúa Kito', 'Category': 'Khám phá', 'Cost': 50000},
        {'Destination': 'Vũng Tàu', 'Name': 'Bãi Sau', 'Category': 'Thư giãn', 'Cost': 100000}
    ]
    df = pd.DataFrame(data)
    df.to_csv('places_db.csv', index=False)
    print("✅ Hoàn tất! Đã tạo file places_db.csv thành công.")

if __name__ == "__main__":
    generate_database()