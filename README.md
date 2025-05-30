# Embedding dữ liệu  từ file txt

**Tạo môi trường conda**

```
conda create -n embedding python=3.10
```

**Activate môi trường conda**

```
conda activate embedding
```

**Cài đặt các gói python**

```
pip install -r requirement.langchain.txt
```

**Đi đến đường dẫn vừa clone source code**

```
cd /path/to/your/project
```

**Chạy file**

```
python embedding.py
```



# Chatbot

**Tạo môi trường conda**

```
conda create -n chatbot python=3.10
```

**Activate môi trường conda**

```
conda activate chatbot
```

**Đi đến đường dẫn vừa clone source code**

```
cd /path/to/your/project
```

**Cài đặt các gói python**

```
pip install -r requirement.langchain.txt
```

**Chạy file**

```
python -m src.chat.chatbot
```

**Link fix lỗi vm.max_map_count trong docker khi chạy opensearch**

```
https://stackoverflow.com/questions/42889241/how-to-increase-vm-max-map-count
```
