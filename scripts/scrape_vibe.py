import os
import pandas as pd
from bing_image_downloader import downloader

# 1. Cấu hình đường dẫn (Vì script nằm trong thư mục 'scripts' nên trỏ ngược ra ngoài)
csv_path = r'D:\Travel_AI_best\data\places_db.csv' 
output_dir = r'D:\Travel_AI_best\data\vibe_dataset'

def main():
    print("Bắt đầu đọc dữ liệu từ CSV...")
    
    # 2. Đọc file CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {csv_path}. Bạn kiểm tra lại đường dẫn nhé.")
        return

    # 3. Lặp qua từng dòng để tải ảnh
    for index, row in df.iterrows():
        place_name = row['Name']
        destination = row['Destination']
        
        # Ghép Tên và Địa điểm để search chính xác (VD: "Cộng Cà Phê Sapa")
        search_query = f"{place_name} {destination}"
        print(f"\n[{index + 1}/{len(df)}] Đang tải ảnh cho: {search_query}")
        
        # Gọi hàm tải ảnh của bing-image-downloader
        try:
            downloader.download(
                search_query,
                limit=15,               # Số lượng ảnh cần tải cho mỗi quán
                output_dir=output_dir,  # Thư mục gốc chứa data
                adult_filter_off=True,  # Tắt filter để tìm kiếm rộng hơn
                force_replace=False,    # Không tải đè nếu thư mục đã có ảnh
                timeout=60,             # Thời gian chờ tối đa
                verbose=False           # Tắt các log lặt vặt của thư viện
            )
        except Exception as e:
            print(f"Có lỗi khi tải {search_query}: {e}")

    # 4. Tạo sẵn thư mục noise_others để bạn tự thêm "ảnh tào lao" sau
    noise_dir = os.path.join(output_dir, 'noise_others')
    os.makedirs(noise_dir, exist_ok=True)
    
    print("\n=========================================")
    print("HOÀN THÀNH CÀO DATA!")
    print(f"Toàn bộ ảnh đã được lưu gọn gàng tại: {output_dir}")
    print("Nhớ chép thêm vài ảnh chó, mèo, xe cộ... vào thư mục 'noise_others' nhé!")
    print("=========================================")

if __name__ == "__main__":
    main()