import pickle
import warnings
warnings.filterwarnings('ignore')
from nltk.corpus import stopwords
import nltk
import tkinter as tk
from tkinter import font as tkfont

# Download NLTK data if not already present
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Text cleaning function (must match training preprocessing)
def clean_review(review, stp_words):
    clean_review = " ".join(word.lower() for word in review.split()
                            if word.lower() not in stp_words and word.isalpha())
    return clean_review

# Load the saved model
def load_model():
    with open('sentiment_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['vectorizer'], data['classes']


# ── GUI ──────────────────────────────────────────────────────────────────────

COLORS = {
    'bg':       '#0f0f0f',
    'surface':  '#1a1a1a',
    'border':   '#2a2a2a',
    'accent':   '#e8c547',
    'text':     '#f0f0f0',
    'muted':    '#666666',
    'negative': '#e05252',
    'neutral':  '#e8c547',
    'positive': '#52c07a',
}

SENTIMENT_COLORS = {
    'Negative': COLORS['negative'],
    'Neutral':  COLORS['neutral'],
    'Positive': COLORS['positive'],
}

SENTIMENT_ICONS = {
    'Negative': '✕',
    'Neutral':  '◎',
    'Positive': '✓',
}


class SentimentApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Product Review Sentiment Analyzer')
        self.configure(bg=COLORS['bg'])
        self.resizable(False, False)

        # Center window
        w, h = 560, 500
        self.geometry(f'{w}x{h}+{(self.winfo_screenwidth()-w)//2}+{(self.winfo_screenheight()-h)//2}')

        # Load model
        self.model, self.vectorizer, self.classes = load_model()
        self.stp_words = set(stopwords.words('english'))

        self._build_ui()

    def _build_ui(self):
        # Fonts
        title_font   = tkfont.Font(family='Courier', size=13, weight='bold')
        label_font   = tkfont.Font(family='Courier', size=9)
        input_font   = tkfont.Font(family='Courier', size=11)
        result_font  = tkfont.Font(family='Courier', size=22, weight='bold')
        conf_font    = tkfont.Font(family='Courier', size=10)
        btn_font     = tkfont.Font(family='Courier', size=10, weight='bold')

        pad = dict(padx=32)

        # Title
        tk.Label(self, text='PRODUCT REVIEW SENTIMENT ANALYZER', bg=COLORS['bg'],
                 fg=COLORS['accent'], font=title_font).pack(pady=(28, 2), **pad, anchor='w')

        tk.Frame(self, bg=COLORS['accent'], height=1).pack(fill='x', **pad)

        # Input label
        tk.Label(self, text='ENTER REVIEW HERE: ', bg=COLORS['bg'],
                 fg=COLORS['muted'], font=label_font).pack(**pad, anchor='w')

        # Text area frame
        text_frame = tk.Frame(self, bg=COLORS['border'], bd=0)
        text_frame.pack(fill='x', **pad, pady=(4, 0))

        self.text_input = tk.Text(
            text_frame, height=6, font=input_font,
            bg=COLORS['surface'], fg=COLORS['text'],
            insertbackground=COLORS['accent'],
            relief='flat', bd=8,
            wrap='word', spacing1=2,
        )
        self.text_input.pack(fill='x')
        self.text_input.bind('<Return>', self._on_enter)

        # Character hint
        self.hint = tk.Label(self, text='press Enter or click Analyze',
                             bg=COLORS['bg'], fg=COLORS['muted'], font=label_font)
        self.hint.pack(**pad, anchor='e', pady=(3, 0))

        # Analyze button
        self.btn = tk.Button(
            self, text='ANALYZE →', font=btn_font,
            bg=COLORS['accent'], fg=COLORS['bg'],
            activebackground='#d4b03a', activeforeground=COLORS['bg'],
            relief='flat', bd=0, padx=20, pady=8, cursor='hand2',
            command=self.analyze,
        )
        self.btn.pack(**pad, anchor='w', pady=(14, 0))

        # Divider
        tk.Frame(self, bg=COLORS['border'], height=1).pack(fill='x', **pad, pady=(20, 0))

        # Result area
        result_frame = tk.Frame(self, bg=COLORS['bg'])
        result_frame.pack(fill='x', **pad, pady=(16, 0))

        self.icon_label = tk.Label(result_frame, text='—', bg=COLORS['bg'],
                                   fg=COLORS['muted'], font=tkfont.Font(family='Courier', size=28, weight='bold'))
        self.icon_label.pack(side='left')

        right = tk.Frame(result_frame, bg=COLORS['bg'])
        right.pack(side='left', padx=(14, 0))

        self.result_label = tk.Label(right, text='awaiting input', bg=COLORS['bg'],
                                     fg=COLORS['muted'], font=result_font)
        self.result_label.pack(anchor='w')

        self.conf_label = tk.Label(right, text='', bg=COLORS['bg'],
                                   fg=COLORS['muted'], font=conf_font)
        self.conf_label.pack(anchor='w')

    # ── Logic ────────────────────────────────────────────────────────────────

    def _on_enter(self, event):
        # Shift+Enter = newline, plain Enter = analyze
        if not event.state & 0x1:
            self.analyze()
            return 'break'

    def analyze(self):
        review = self.text_input.get('1.0', 'end').strip()
        if not review:
            self.result_label.config(text='enter a review', fg=COLORS['muted'])
            self.conf_label.config(text='')
            self.icon_label.config(text='—', fg=COLORS['muted'])
            return

        cleaned    = clean_review(review, self.stp_words)
        review_vec = self.vectorizer.transform([cleaned]).toarray()

        prediction = self.model.predict(review_vec)[0]
        confidence = self.model.predict_proba(review_vec).max()
        sentiment  = self.classes[prediction + 1]   # +1 because classes are -1,0,1

        color = SENTIMENT_COLORS[sentiment]
        icon  = SENTIMENT_ICONS[sentiment]

        self.result_label.config(text=sentiment.upper(), fg=color)
        self.conf_label.config(text=f'confidence  {confidence:.1%}', fg=COLORS['muted'])
        self.icon_label.config(text=icon, fg=color)


if __name__ == '__main__':
    app = SentimentApp()
    app.mainloop()