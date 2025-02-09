import pandas as pd
import numpy as np
from datetime import datetime

def create_test_data(output_file: str = "test_data.xlsx"):
    """Create a test Excel file with sample social media comments based on Vodafone data structure."""
    
    # Sample data matching the structure from the spreadsheet
    data = {
        'Platform': [],
        'Post/comments': [],
        'Sentiment': [],
        'Date': [],
        'Number of shares/retweets': [],
        'User/account': [],
        'No of followers': [],
        'Link': []
    }
    
    # Sample comments with different scenarios
    sample_data = [
        # Customer service issues
        {
            'platform': 'facebook',
            'comment': 'معايا خط فودافون شركات جبت جواب من المفوض وديته فرع ابو قير الخط لسه ماشتغلش الخط بيستقبل بس',
            'sentiment': 'negative',
            'shares': 15,
            'user': 'customer123',
            'followers': 250
        },
        # Positive feedback
        {
            'platform': 'twitter',
            'comment': '@VodafoneEgypt شكرا ع الرد والاهتمام بعت رقمى ف ال DM',
            'sentiment': 'positive',
            'shares': 5,
            'user': '@happy_customer',
            'followers': 1200
        },
        # Service complaints
        {
            'platform': 'facebook',
            'comment': 'بقالي اكتر من 15 يوم هدايا فودافون كاش اللي هي الكاش باك والوحدات متوقفه بدون سبب',
            'sentiment': 'negative',
            'shares': 30,
            'user': 'concerned_user',
            'followers': 500
        },
        # Neutral inquiry
        {
            'platform': 'twitter',
            'comment': '@VodafoneEgypt متى يتم تفعيل الخدمة الجديدة؟',
            'sentiment': 'neutral',
            'shares': 2,
            'user': '@curious_user',
            'followers': 800
        },
        # Technical issue
        {
            'platform': 'facebook',
            'comment': 'السلام عليكم يا جماعه محدش يحول رصيد او فودافون كاش من فوري لفودافون فلوسي ضاعت',
            'sentiment': 'negative',
            'shares': 45,
            'user': 'tech_issue',
            'followers': 350
        }
    ]
    
    # Generate multiple entries from sample data
    for _ in range(3):  # Repeat each sample 3 times with variations
        for entry in sample_data:
            data['Platform'].append(entry['platform'])
            data['Post/comments'].append(entry['comment'])
            data['Sentiment'].append(entry['sentiment'])
            data['Date'].append(datetime.now().strftime('%Y-%m-%dT%H:%M:%S+0000'))
            data['Number of shares/retweets'].append(entry['shares'] + np.random.randint(-5, 15))
            data['User/account'].append(entry['user'] + str(np.random.randint(100, 999)))
            data['No of followers'].append(entry['followers'] + np.random.randint(-100, 100))
            data['Link'].append(f"https://{'twitter' if entry['platform'] == 'twitter' else 'facebook'}.com/{np.random.randint(100000, 999999)}")
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to Excel
    df.to_excel(output_file, index=False)
    print(f"Test data saved to {output_file}")
    return output_file

if __name__ == "__main__":
    create_test_data() 