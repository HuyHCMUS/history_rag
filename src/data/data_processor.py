# Data processing logic
import pandas as pd
import re
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize
from embeddings import create_embeddings
from tqdm import tqdm
from datetime import datetime

def extract_years(text):
    pattern = r'\[(-?\d{2,4})\s*TCN?\]|\[(-?\d{2,4})\]|/(-?\d{2,4})/'
    matches = re.findall(pattern, text)
    
    years = []
    for match in matches:
        if match[0]:  # Năm có "TCN", lưu với dấu âm
            years.append(f"-{match[0]}")
        elif match[1]:  # Năm bình thường trong []
            years.append(match[1])
        elif match[2]:  # Năm trong //
            years.append(match[2])
    if years: return years[0]
    else: return None

def classify_event(text):
    """
    Phân loại sự kiện trong văn bản dựa trên các từ khóa.
    
    Args:
        text (str): Văn bản đầu vào.
    
    Returns:
        list: Danh sách các loại sự kiện tìm thấy trong văn bản.
    """
    categories = {
    "Ngoại giao": ["sai sứ", "sang cống", "đáp lễ", "sang phong", "sang chầu", "biếu nhà", "liên minh", "giao hảo", "nghị hòa", "hòa ước", "triều cống", "giao thương", "tiếp sứ", "hội kiến", "sứ thần", "kết thông gia"],
    "Chính trị": ["đổi niên hiệu", "hạ lệnh", "xuống chiếu", "sắc chỉ", "phong", "niên hiệu", "ra lệnh", "luật", "lên ngôi", "phong vua", "tấu lên", "sai quan", "bãi nhiệm", "cải cách", "sắc lệnh","phong tước","thái tử", "công chúa","hoàng hậu", "thái hậu"],
    "Văn hóa": ["thi hội", "đua thuyền", "chùa", "đạo", "đền", "miếu", "lễ", "học", "trụ trì", "diễn", "mở hội", "hát tuồng", "múa rối", "thư pháp", "văn bia"],
    "Giáo dục": [ "kinh sử","lấy đỗ", "lấy sinh", "học sinh", "người đỗ", "tuyển chọn nhân tài", "khoa cử", "mở trường", "dạy học", "tiến sĩ", "trạng nguyên", "bảng vàng","sách", "in sách", "dịch sách"],
    "Quân sự": ["đi đánh", "làm phản", "làm loạn", "đánh chiếm", "tướng quân", "đem quân", "giặc", "trận", "chiến thắng", "thua", "tấn công", "quân giặc", "quân đánh", "bắt sống", "chiếm đóng", "vây thành", "phục kích", "bố trận", "đạo quân", "đóng giữ"],
    "Thời tiết": ["hạn", "lụt", "bão", "động đất", "sao chổi", "sấm", "mưa", "dịch", "nhật thực", "nguyệt thực", "nước to", "mất mùa", "nắng gắt", "tuyết", "băng giá"],
    "Đối thoại": ["nói", "rằng", "làm thơ", "nghị luận", "tranh biện"],
    "Giới thiệu nhân vật": ["tên huý là", "ở ngôi", "cha là", "xuất thân", "tôn xưng", "danh hiệu"],
    "Kinh tế - Tài chính": ["bạc", "vàng", "tiền đồng", "phát hành", "thuế", "miễn thuế", "định giá", "giá cả", "buôn bán", "thương mại" ,],
    "Xã hội - Đời sống": ["cứu tế", "phát chẩn", "y tế", "bệnh dịch", "thuốc", "chữa bệnh", "cứu đói", "trợ cấp", "giảm tô", "giảm thuế","cày","ruộng đất", "kho lúa"],
    "Kiến trúc - Xây dựng": ["xây thành", "đắp đê", "làm cầu", "sửa chùa", "xây cung", "làm đường", "dựng bia", "khắc chữ","xây chùa"],
    "Khoa học - Kỹ thuật": ["sao trời", "thiên văn", "tính lịch", "đo đạc", "y thuật", "phát minh"]
}
    
    event_types = []
    text = text.lower()  
    
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in text:
                event_types.append(category)
                break  # Chỉ cần tìm thấy 1 từ khóa là đủ để phân loại vào nhóm đó

    return event_types

def extract_name(text):   
    regex = r'[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+(?:\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]+)+'
    
    name_raw = re.findall(regex, text)
    
    name_result = []
    for name in name_raw:
        # Loại bỏ dấu câu ở cuối nếu có
        name_final = re.sub(r'[\.,!?;:]$', '', name)
        name_result.append(name_final)
    
    return set(name_result)

trieudai_vn = [
    "Nhà Đinh", "nhà Đinh", "Vua Đinh", "vua Đinh", 
    "Nhà Tiền Lê", "nhà Tiền Lê", "Vua Lê", "vua Lê",  
    "Nhà Lê", "nhà Lê", 
    "Nhà Lý", "nhà Lý", "Vua Lý", "vua Lý",  
    "Nhà Trần", "nhà Trần", "Vua Trần", "vua Trần",  
    "Nhà Hồ", "nhà Hồ", "Vua Hồ", "vua Hồ",  
    "Nhà Hậu Lê", "nhà Hậu Lê",
    "Nhà Nguyễn", "nhà Nguyễn", "Vua Nguyễn", "vua Nguyễn"
]

# Triều đại Trung Quốc (có thể gọi bằng "Nhà", "quân", "giặc")
trieudai_tq = [
    "Nhà Tần", "nhà Tần", "quân Tần", "giặc Tần",  
    "Nhà Hán", "nhà Hán", "quân Hán", "giặc Hán",  
    "Nhà Đường", "nhà Đường", "quân Đường", "giặc Đường", 'vua Đường',  
    "Nhà Tống", "nhà Tống", "quân Tống", "giặc Tống", 'vua Tống',
    "Nhà Nguyên", "nhà Nguyên", "quân Nguyên", "giặc Nguyên",  'vua Nguyên', 
    "Nhà Minh", "nhà Minh", "quân Minh", "giặc Minh", 'vua Minh',
    "Nhà Thanh", "nhà Thanh", "quân Thanh", "giặc Thanh"
]

trieudai = set(trieudai_vn + trieudai_tq)

keywords = set([
    "sai sứ", "sang cống", "đáp lễ", "sang phong", "sang chầu", "biếu nhà", "liên minh", "giao hảo", "nghị hòa", "hòa ước", "triều cống", "giao thương", "tiếp sứ", "hội kiến", "sứ thần", "kết thông gia",  # Ngoại giao
    "đổi niên hiệu", "hạ lệnh", "xuống chiếu", "sắc chỉ", "phong", "niên hiệu", "ra lệnh", "luật", "lên ngôi", "phong vua", "tấu lên", "sai quan", "bãi nhiệm", "cải cách", "sắc lệnh","phong tước","thái tử", "công chúa","hoàng hậu", "thái hậu",  # Chính trị
    "thi hội", "đua thuyền", "chùa", "đạo", "đền", "miếu", "lễ", "học", "trụ trì", "diễn", "mở hội", "hát tuồng", "múa rối", "thư pháp", "văn bia",  # Văn hóa
    "kinh sử","lấy đỗ", "lấy sinh", "học sinh", "người đỗ", "tuyển chọn nhân tài", "khoa cử", "mở trường", "dạy học", "tiến sĩ", "trạng nguyên", "bảng vàng","sách", "in sách", "dịch sách",  # Giáo dục
    "đi đánh", "làm phản", "làm loạn", "đánh chiếm", "tướng quân", "đem quân", "giặc", "trận", "chiến thắng", "thua", "tấn công", "quân giặc", "quân đánh", "bắt sống", "chiếm đóng", "vây thành", "phục kích", "bố trận", "đạo quân", "đóng giữ",  # Chiến tranh
    "hạn", "lụt", "bão", "động đất", "sao chổi", "sấm", "mưa", "dịch", "nhật thực", "nguyệt thực", "nước to", "mất mùa", "nắng gắt", "tuyết", "băng giá",  # Thời tiết
    "nói", "rằng", "làm thơ", "nghị luận", "tranh biện",  # Đối thoại
    "tên huý là", "ở ngôi", "cha là", "xuất thân", "tôn xưng", "danh hiệu",
    "bạc", "vàng", "tiền đồng", "phát hành", "thuế", "miễn thuế", "định giá", "giá cả", "buôn bán", "thương mại" ,
    "cứu tế", "phát chẩn", "y tế", "bệnh dịch", "thuốc", "chữa bệnh", "cứu đói", "trợ cấp", "giảm tô", "giảm thuế","cày","ruộng đất", "kho lúa",
    "xây thành", "đắp đê", "làm cầu", "sửa chùa", "xây cung", "làm đường", "dựng bia", "khắc chữ","xây chùa",
    "sao trời", "thiên văn", "tính lịch", "đo đạc", "y thuật", "phát minh"
])

def extract_tags(text):
    text_lower = text.lower()
    named_entities = extract_name(text)
    found_dynasties = {dyn for dyn in trieudai if dyn in text}
    event_keywords = {kw for kw in keywords if kw in text_lower}
    
    return event_keywords | named_entities | found_dynasties

def setup_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

# Hàm tạo chunk thông minh dựa trên các câu hoàn chỉnh
def create_chunks(text, max_chunk_size=200, min_chunk_size=50):
    """
    Tạo các chunk thông minh dựa trên ranh giới câu và đoạn.
    
    Args:
        text (str): Văn bản đầu vào.
        max_chunk_size (int): Số từ tối đa trong một chunk.
        min_chunk_size (int): Số từ tối thiểu để tạo thành một chunk độc lập.
    
    Returns:
        list: Danh sách các chunk có nội dung ngữ nghĩa hợp lý.
    """
    setup_nltk()
    
    # Tách đoạn văn (paragraph)
    paragraphs = re.split(r'\n\s*\n', text)
    chunks = []
    current_chunk = []
    current_chunk_size = 0
    
    for paragraph in paragraphs:
        # Bỏ qua đoạn trống
        if not paragraph.strip():
            continue
            
        # Tách câu trong đoạn
        sentences = sent_tokenize(paragraph)
        
        paragraph_added = False
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_length = len(sentence_words)
            
            # Trường hợp 1: Câu quá dài, buộc phải tách ra thành chunk riêng
            if sentence_length > max_chunk_size:
                # Xử lý chunk hiện tại nếu có
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_chunk_size = 0
                
                # Tách câu dài thành nhiều chunk
                for i in range(0, sentence_length, max_chunk_size):
                    if i + max_chunk_size >= sentence_length:
                        # Phần cuối của câu dài
                        if current_chunk_size == 0:
                            # Thêm trực tiếp vào chunks nếu chưa có gì trong current_chunk
                            chunks.append(' '.join(sentence_words[i:]))
                        else:
                            # Thêm vào chunk hiện tại
                            current_chunk.extend(sentence_words[i:])
                            current_chunk_size += len(sentence_words[i:])
                    else:
                        # Phần giữa của câu dài
                        chunks.append(' '.join(sentence_words[i:i+max_chunk_size]))
                        
            # Trường hợp 2: Câu vừa + chunk hiện tại sẽ vượt quá max_chunk_size
            elif current_chunk_size + sentence_length > max_chunk_size:
                # Hoàn thành chunk hiện tại
                chunks.append(' '.join(current_chunk))
                
                # Bắt đầu chunk mới với câu hiện tại
                current_chunk = sentence_words
                current_chunk_size = sentence_length
                paragraph_added = True
                
            # Trường hợp 3: Thêm câu vào chunk hiện tại
            else:
                current_chunk.extend(sentence_words)
                current_chunk_size += sentence_length
                paragraph_added = True
        
        # Kết thúc đoạn, thêm dấu xuống dòng nếu không phải là đoạn cuối
        if paragraph_added and current_chunk:
            current_chunk[-1] = current_chunk[-1] + '.'
            
    # Xử lý chunk cuối cùng nếu có
    if current_chunk:
        # Nếu chunk cuối quá nhỏ và có chunk trước đó, gộp với chunk cuối
        if current_chunk_size < min_chunk_size and len(chunks) > 0:
            last_chunk = chunks[-1].split()
            combined_size = len(last_chunk) + current_chunk_size
            
            if combined_size <= max_chunk_size * 1.2:  # Cho phép vượt một chút
                chunks[-1] = ' '.join(last_chunk + current_chunk)
            else:
                chunks.append(' '.join(current_chunk))
        else:
            chunks.append(' '.join(current_chunk))
    
    return chunks

def process_dataset(data, chunk_size=200, model_name="vinai/phobert-base-v2"):
    """
    Xử lý toàn bộ dataset lịch sử để chuẩn bị cho RAG.
    
    Args:
        data (list): Dataset chứa các record lịch sử.
        chunk_size (int): Kích thước tối đa của mỗi đoạn.
        model_name (str): Tên của mô hình sentence transformer.
    
    Returns:
        dict: Dữ liệu đã được xử lý và tạo embeddings.
    """
    # Khởi tạo các list để lưu trữ kết quả
    all_chunks = []
    all_metadata = []
    original_indices = []
        
    setup_nltk()
    records = data
    print(data)
    print('###########')
    # Xử lý từng record
    current_year = '-258'
    for idx, record in tqdm(enumerate(records), total=len(records), desc="Processing records"):
        text = record
        if not text:
            continue
        
        # Trích xuất metadata
        event_types = classify_event(text)
        tags = extract_tags(text)
        years = extract_years(text)
        
        if years == None:
            years = current_year
        else:
            current_year = years
        if not tags:
            tags = ['lịch sử']
        if not event_types:
            event_types = ['lịch sử']
        
        # Tạo chunk từ văn bản
        chunks = create_chunks(text, chunk_size)
        # Thêm thông tin vào danh sách
        for chunk in chunks:
            all_chunks.append(chunk)
            metadata = {
                'original_index': idx,
                'event_types': event_types,
                'tags': list(tags),  # Chuyển set thành list để dễ serialize
                'years': years,
                'chunk_text': chunk
            }
            
            all_metadata.append(metadata)
            original_indices.append(idx)
    
    # Tạo embeddings
    print("Đang tạo embeddings...")
    embeddings = create_embeddings(all_chunks, model_name)
    
    # Đóng gói kết quả
    result = {
        'chunks': all_chunks,
        'metadata': all_metadata,
        #'embeddings': embeddings,
        'original_indices': original_indices,
        #'embedding_model': model_name,
        #'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'chunk_size': chunk_size,
        #'overlap': overlap
    }
    print(result)
    return result
