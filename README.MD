## Basic UI for Vision Model
---

**Step 1: Clone Repo**
```bash
git clone https://github.com/Anthony-Lionetti/streamlit-vision-ui.git
```

**Step 2: Create a Virtual Environment**\
(windows)
```bash
python -m venv ds-vision-env
ds-vision-env/Scripts/activate
```

(mac/linux)
```bash
python -m venv ds-vision-env
source ds-vision-env/bin/activate
```

**Step 3: Install Dependencies**
```bash
pip install streamlit pandas fastai
```

**Step 4: Load Model File**\
Add your trained model file into the root folder directory

**Step 5: Edit Variables**\
In the `app.py` file update `MODEL_PATH` and `ITEM` to your specific project

**Step 5: Start local server**
```bash
streamlit run app.py
```