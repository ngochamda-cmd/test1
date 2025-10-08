import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính 📊")

# --- Khởi tạo Session State cho Chat ---
if "messages" not in st.session_state:
    # Lưu trữ lịch sử tin nhắn
    st.session_state["messages"] = []
if "chat_session" not in st.session_state:
    # Lưu trữ phiên chat Gemini để duy trì ngữ cảnh
    st.session_state["chat_session"] = None

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý lỗi chia cho 0 khi tính tỷ trọng
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Nhận xét Ban đầu (Không lưu trạng thái) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét ban đầu."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Khởi tạo giá trị mặc định để tránh lỗi
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, kiểm tra chia cho 0
                if no_ngan_han_N != 0:
                    thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                if no_ngan_han_N_1 != 0:
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else "N/A"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else "N/A",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float) else None
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            
            # Chuẩn bị dữ liệu để gửi cho AI (Dùng cho cả Chức năng 5 và 6)
            tsnh_tang_truong = "N/A"
            try:
                tsnh_tang_truong = f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%"
            except IndexError:
                pass
                
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    tsnh_tang_truong, 
                    f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, float) else "N/A", 
                    f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) else "N/A"
                ]
            }).to_markdown(index=False) 

            # --- Chức năng 5: Nhận xét AI và Khởi tạo Chat ---
            st.subheader("5. Nhận xét AI & Khởi tạo Chat Session")
            
            if st.button("Yêu cầu AI Phân tích & Kích hoạt Chat"):
                api_key = st.secrets.get("GEMINI_API_KEY")

                if api_key:
                    # 1. Thực hiện phân tích ban đầu (Nhận xét)
                    with st.spinner('Đang gửi dữ liệu, chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)

                    st.markdown("**Kết quả Phân tích ban đầu từ Gemini AI:**")
                    st.info(ai_result)

                    # 2. Thiết lập Chat Session với Context (LƯU VÀO SESSION STATE)
                    try:
                        client = genai.Client(api_key=api_key)
                        
                        # Đặt ngữ cảnh cho AI trong suốt phiên chat
                        chat_system_instruction = f"""
                        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp.
                        Dữ liệu tài chính đã phân tích chi tiết mà bạn cần tham khảo cho mọi câu hỏi tiếp theo là:
                        {data_for_ai}
                        
                        Hãy trả lời các câu hỏi của người dùng dựa trên dữ liệu này. Nếu câu hỏi không liên quan đến tài chính hoặc dữ liệu được cung cấp, hãy lịch sự từ chối và yêu cầu họ hỏi về dữ liệu đã tải lên.
                        """
                        
                        st.session_state["chat_session"] = client.chats.create(
                            model='gemini-2.5-flash',
                            system_instruction=chat_system_instruction
                        )
                        
                        # Khởi tạo lịch sử chat với kết quả phân tích ban đầu
                        st.session_state["messages"] = [
                            {"role": "assistant", "content": ai_result}
                        ]
                        st.success("Thiết lập chat thành công! Hãy kéo xuống bước 6 để bắt đầu hỏi đáp chuyên sâu.")
                    except APIError as e:
                        st.error(f"Lỗi khởi tạo Chat Session: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}")
                        st.session_state["chat_session"] = None # Reset session on failure

                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

            # --- Chức năng 6: Khung Chat Hỏi Đáp Thêm với Gemini ---
            st.subheader("6. Hỏi đáp chuyên sâu với AI (Duy trì Ngữ cảnh)")

            if st.session_state["chat_session"]:
                # Hiển thị lịch sử chat
                for message in st.session_state["messages"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Xử lý input từ người dùng
                if prompt := st.chat_input("Hỏi AI về báo cáo tài chính này..."):
                    
                    # 1. Thêm tin nhắn người dùng vào lịch sử và hiển thị
                    st.session_state["messages"].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # 2. Gửi câu hỏi đến phiên chat đang hoạt động
                    with st.spinner("Đang gửi câu hỏi và chờ câu trả lời từ AI..."):
                        try:
                            # Phiên chat sẽ tự động duy trì ngữ cảnh (dữ liệu tài chính)
                            response = st.session_state["chat_session"].send_message(prompt)
                            ai_response = response.text
                            
                            # 3. Thêm phản hồi của AI vào lịch sử và hiển thị
                            st.session_state["messages"].append({"role": "assistant", "content": ai_response})
                            with st.chat_message("assistant"):
                                st.markdown(ai_response)
                        
                        except APIError as e:
                            error_msg = f"Lỗi gọi Gemini API trong Chat: {e}"
                            st.error(error_msg)
                            st.session_state["messages"].append({"role": "assistant", "content": error_msg})

            else:
                st.info("Nhấn **'Yêu cầu AI Phân tích & Kích hoạt Chat'** ở bước 5 để thiết lập ngữ cảnh (dữ liệu tài chính) và mở khung chat hỏi đáp.")


    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
