# Assignment-2-NLP-vs-GenAI
Assignment-2-NLP-vs-GenAI
NLP 作業二：傳統 NLP 與 現代 GenAI 方法實作比較
本專案為自然語言處理 (NLP) 課程作業，旨在透過實作探討「傳統統計式 NLP 技術」與「現代生成式 AI (GenAI)」在文本處理任務上的效能與特性差異。

專案內容包含手動實作 TF-IDF 演算法、規則式分類器，並串接 Google Gemini API 進行 Embeddings 計算、零樣本分類 (Zero-shot Classification) 與生成式摘要，最終產出詳細的效能比較報告。

📂 專案結構
本專案包含三個獨立執行的 Python 腳本與相關設定檔：

tradtional_methods.py: 傳統 NLP 方法實作
手動實作 TF-IDF 演算法與 Cosine Similarity。
實作規則式 (Rule-based) 文本分類器。
實作統計式 (詞頻基礎) 自動摘要。
產出: tfidf_similarity_matrix.csv

modern_method.py: 現代 AI 方法實作
使用 Google Gemini API 計算文本 Embeddings。
利用 LLM 進行含信心分數的文本分類。
使用生成式 AI 進行摘要改寫。
產出: classification_result.csv

comparsion.py: 比較分析與效能評測
針對相似度、分類、摘要三項任務進行執行時間與效能的自動化評測。
生成比較分析表格與數據。
產出: performance_metrics.json

💻 執行說明
請依序執行以下指令：

步驟 1：執行 Part A (傳統方法)

  此步驟不需要 API Key，將使用本機運算。

  python tradtional_methods.py

  執行結果：終端機將顯示 TF-IDF 關鍵詞分析、規則分類結果，並於目錄下生成 tfidf_similarity_matrix.csv。


步驟 2：執行 Part B (現代 AI)

  此步驟需要有效的 API Key。

  python modern_method.py

  執行結果：將呼叫 Gemini API 進行 Embeddings 計算與分類，並於目錄下生成 classification_result.csv (包含情感、主題與信心分數)。


步驟 3：執行 Part C (評測報告)

  此步驟進行效能計時與比較。

  python part_c.py

  執行結果：終端機將顯示完整的比較分析表格，並於目錄下生成 performance_metrics.json (包含詳細的時間與效能數據)。


📊 產出檔案說明
執行完畢後，您將獲得以下檔案：

tradtional_methods：文件間的 TF-IDF 餘弦相似度矩陣，產出tfidf_similarity_matrix.csv。

modern_method：包含原始文本、AI 判讀之情感、主題及信心分數的表格，產出classification_result.csv。

comparsion：紀錄傳統方法與 GenAI 在各任務上的處理時間與評測指標，產出performance_metrics.json。
