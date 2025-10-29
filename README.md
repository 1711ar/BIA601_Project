 BIA601 – GA Feature Selection Web App

 اختيار الميزات بخوارزمية جينية (GA) مع مقارنات تقليدية (PCA, SelectKBest). يهدف لتقليل عدد الأعمدة مع الحفاظ على الأداء

المزايا
- رفع CSV/Excel وتحديد عمود الهدف (تصنيف/انحدار).
- اختيار ميزات بـ GA (قناع ثنائي، نخبوية، تزاوج، طفرة، إيقاف مبكر).
- مقارنات Baselines: All-features / PCA / SelectKBest.
- عرض النتائج (CV score، عدد الميزات) وتنزيل البيانات المختزلة.
- تنفيذ سريع وإعدادات قابلة للتحكم من الواجهة.
 التشغيل محليا (Windows / Linux / macOS)
bash
pip install -r requirements.txt
streamlit run webapp_streamlit.py
