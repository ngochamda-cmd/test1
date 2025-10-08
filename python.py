import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError
import time 

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính & Chatbot AI 📊")

# =========================================================================
# --- THÊM CSS TÙY CHỈNH CHO KHUNG CHAT (Đỏ và Trắng) ---
# Sử dụng CSS Injection để ghi đè màu sắc mặc định của bong bóng chat
# =========================================================================
custom_css = """
<style>
/* AI/Assistant Message Bubble (Đỏ đậm, Chữ Trắng) */
/* Selector nhắm vào khối tin nhắn với role="assistant" */
[data-testid="stChatMessage"][data-message-role="assistant"] > div:nth-child(2) > div:nth-child(1) {
    background-color: #B71C1C !important; /* Đỏ đậm */
    border: none !important;
    color: white !important;
}

/* User Message Bubble (Trắng, Chữ Đen) */
/* Selector nhắm vào khối tin nhắn với role="user" */
[data-testid="stChatMessage"][data-message-role="user"] > div:nth-child(2) > div:nth-child(1) {
    background-color: white !important; /* Trắng */
    border: 1px solid #B71C1C !important; /* Viền Đỏ cho nổi bật */
    color: black !important;
}

/* Đảm bảo nội dung Markdown (chữ) bên trong bong bóng có màu phù hợp */
[data-testid="stChatMessage"][data-message-role="assistant"] .stMarkdown {
    color: white !important;
}
[data-testid="stChatMessage"][data-message-role="user"] .stMarkdown {
    color: black !important;
}

/* Điều chỉnh icon/nút bên trong bong bóng đỏ (ví dụ: nút Copy) */
[data-testid="stChatMessage"][data-message-role="assistant"] button {
    color: white !important; 
    border-color: white !important;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# --- 1. Khởi tạo State và API Key (Đảm bảo chỉ chạy 1 lần) ---
# Khởi tạo session state cho lịch sử chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Lấy API key và khởi tạo Client
client = None
MODEL_NAME = 'gemini-2.5-flash'
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
    client = genai.Client(api_key=API_KEY)
except KeyError:
    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
except Exception as e:
    st.error(f"Lỗi khởi tạo Gemini Client: {e}")


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
    
    # Xử lý chia cho 0
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Phân tích tự động (Chức năng 5) ---
def get_ai_analysis(data_for_ai, client, model_name):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    if not client:
        return "Lỗi: Không thể kết nối với Gemini API do thiếu Khóa API."

    try:
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
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Khởi tạo biến để lưu trữ dữ liệu phân tích dạng chuỗi cho Chatbot
data_for_ai_markdown = None 
thanh_toan_hien_hanh_N = "N/A"
thanh_toan_hien_hanh_N_1 = "N/A"

if uploaded_file is not None:
    try:
        # Tải và tiền xử lý dữ liệu
        df_raw = pd.read_excel(uploaded_file)
        # Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        df_processed = process_financial_data(df_raw.copy())

        # Nếu xử lý thành công
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
            
            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, xử lý chia cho 0 (dùng float('inf') cho trường hợp Nợ = 0)
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else float('inf')
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else float('inf')
                
                col1, col2 = st.columns(2)
                
                value_N_1 = f"{thanh_toan_hien_hanh_N_1:.2f} lần" if thanh_toan_hien_hanh_N_1 != float('inf') else "Không xác định (Nợ = 0)"
                value_N = f"{thanh_toan_hien_hanh_N:.2f} lần" if thanh_toan_hien_hanh_N != float('inf') else "Không xác định (Nợ = 0)"
                delta_value = f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}" if thanh_toan_hien_hanh_N != float('inf') and thanh_toan_hien_hanh_N_1 != float('inf') else None
                
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=value_N_1
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=value_N,
                        delta=delta_value
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"
            
            # --- Chuẩn bị dữ liệu để gửi cho AI và Chatbot (tạo Markdown string) ---
            try:
                # Lấy tốc độ tăng trưởng tài sản ngắn hạn
                tsnh_growth = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]
            except IndexError:
                tsnh_growth = "N/A"
            
            # Chuyển đổi giá trị thanh toán hiện hành sang string cho Markdown
            tt_N_str = f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) and thanh_toan_hien_hanh_N != float('inf') else str(thanh_toan_hien_hanh_N)
            tt_N_1_str = f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, float) and thanh_toan_hien_hanh_N_1 != float('inf') else str(thanh_toan_hien_hanh_N_1)

            data_for_ai_markdown = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{tsnh_growth:.2f}%" if tsnh_growth != "N/A" else "N/A", 
                    tt_N_1_str,
                    tt_N_str
                ]
            }).to_markdown(index=False)

            # --- Chức năng 5: Nhận xét AI Tự động ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI Tự động)")
            
            if st.button("Yêu cầu AI Phân tích"):
                if client:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        # Cập nhật: Sử dụng client và MODEL_NAME đã khởi tạo
                        ai_result = get_ai_analysis(data_for_ai_markdown, client, MODEL_NAME)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Không thể phân tích. Vui lòng kiểm tra Khóa API Gemini.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
    
    # =========================================================================
    # --- PHẦN BỔ SUNG: CHATBOT HỎI ĐÁP VỚI GEMINI (Chức năng 6) ---
    # =========================================================================
    st.markdown("---")
    st.subheader("6. Chatbot Phân tích Tài chính (Hỏi đáp chuyên sâu)")
    st.info("Sử dụng khung chat này để hỏi Gemini về dữ liệu đã tải lên. Ví dụ: 'Tài sản dài hạn có sự thay đổi nào nổi bật không?'")

    if client and data_for_ai_markdown:
        
        # 1. Hiển thị lịch sử chat
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # 2. Xử lý input mới từ người dùng
        if prompt := st.chat_input("Hỏi Gemini về Báo cáo Tài chính của bạn..."):
            
            # Thêm tin nhắn người dùng vào lịch sử Streamlit
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Xây dựng System Instruction (ngữ cảnh)
            system_instruction = f"""
            Bạn là một trợ lý phân tích tài chính chuyên nghiệp. Hãy trả lời câu hỏi của người dùng một cách chính xác, ngắn gọn và dựa trên dữ liệu được cung cấp.

            --- DỮ LIỆU PHÂN TÍCH TÀI CHÍNH ---
            {data_for_ai_markdown}
            --- KẾT THÚC DỮ LIỆU ---
            
            Nếu câu hỏi không liên quan đến dữ liệu tài chính được cung cấp, hãy trả lời một cách hữu ích và duy trì vai trò trợ lý AI.
            """
            
            # Chuẩn bị nội dung gửi đi (System Instruction + Lịch sử Chat)
            # Thêm System Instruction làm tin nhắn đầu tiên để cung cấp ngữ cảnh
            chat_history_with_context = [
                {"role": "user", "parts": [{"text": system_instruction}]}
            ]
            
            # Thêm lịch sử chat vào contents (đảm bảo role là 'user' hoặc 'model')
            for msg in st.session_state.messages:
                # Tránh lặp lại System Instruction nếu đã có trong lịch sử (chỉ lấy role user/assistant)
                role = "model" if msg["role"] == "assistant" else "user"
                # Chỉ thêm nội dung chat, không thêm System Instruction vào chat history
                if msg["content"] != system_instruction: 
                    chat_history_with_context.append({"role": role, "parts": [{"text": msg["content"]}]})

            # Gửi yêu cầu đến Gemini
            with st.chat_message("assistant"):
                with st.spinner("Gemini đang suy nghĩ..."):
                    try:
                        # Sử dụng generate_content_stream để có hiệu ứng gõ
                        response_stream = client.models.generate_content_stream(
                            model=MODEL_NAME,
                            contents=chat_history_with_context
                        )

                        full_response = ""
                        # Hiển thị phản hồi từng phần (streaming effect)
                        response_placeholder = st.empty()
                        for chunk in response_stream:
                            if chunk.text:
                                full_response += chunk.text
                                # Hiển thị con trỏ nhấp nháy cho UX tốt hơn
                                response_placeholder.markdown(full_response + "▌") 
                                
                        response_placeholder.markdown(full_response) # Hiển thị phản hồi hoàn chỉnh
                        
                        # Thêm tin nhắn Gemini vào lịch sử Streamlit
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

                    except APIError as e:
                        error_msg = f"Lỗi gọi Gemini API: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                    except Exception as e:
                        error_msg = f"Đã xảy ra lỗi không xác định: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
    elif client:
        st.warning("Vui lòng tải file để kích hoạt Chatbot phân tích dữ liệu.")
    else:
        st.error("Chatbot không hoạt động do lỗi Khóa API.")
        
    # --- Nút Xóa Lịch sử Chat ---
    if st.session_state.messages and uploaded_file is not None:
        if st.button("Xóa Lịch sử Chat", help="Xóa tất cả các tin nhắn trong khung chat hiện tại."):
            st.session_state.messages = []
            st.experimental_rerun()
        
else:
    # Hiển thị thông tin chờ tải file
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
