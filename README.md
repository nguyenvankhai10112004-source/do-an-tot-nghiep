# Tính năng lượng từ tuyến - demo OpenRouteService

Hướng dẫn nhanh để chạy ứng dụng local:

1. Đăng ký OpenRouteService và lấy `ORS_API_KEY` (https://openrouteservice.org)
2. Đặt biến môi trường trên Windows PowerShell:

```powershell
setx ORS_API_KEY "YOUR_KEY"
$env:ORS_API_KEY = "YOUR_KEY"
```

3. Cài dependencies và chạy server:

```powershell
pip install -r server/requirements.txt
python server/app.py
```

4. Mở trình duyệt đến `http://localhost:5000` — click bản đồ để chọn Start/End, bấm `Lấy tuyến`, rồi `Tính năng lượng`.

Ghi chú:
- Ứng dụng dùng file `code tinh nang luong.py` trong workspace để tính toán năng lượng. Đảm bảo file đó tồn tại ở thư mục gốc của workspace.
- Chỉ có kết quả nhiên liệu ANN được hiển thị. Nhiên liệu vật lý không còn hiển thị.
- ORS có quota miễn phí; tránh gọi elevation cho quá nhiều điểm đồng thời.

## Deploy lên dịch vụ như Railway / Render

1. Đẩy toàn bộ thư mục dự án lên GitHub.
2. Tạo một ứng dụng mới trên Railway hoặc Render.
3. Chỉ định repository và branch của bạn.
4. Nếu cần, đặt command khởi động:
   - `gunicorn server.app:app`
5. Đặt biến môi trường trên dịch vụ:
   - `ORS_API_KEY` = giá trị key của bạn.
6. Nếu dịch vụ yêu cầu, chọn Python runtime 3.13 hoặc 3.11.

Project này đã có các file hỗ trợ:
- `requirements.txt` tại root
- `Procfile`
- `.gitignore`

Sau khi deploy, bạn sẽ nhận được một URL `https://...` để chia sẻ với người khác.
