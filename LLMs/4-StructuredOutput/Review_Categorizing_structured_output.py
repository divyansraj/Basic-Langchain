oneplus_nord_5_reviews = [
    "This phone feels premium in hand with a large 6.83″ AMOLED screen and ultra-smooth 144 Hz refresh rate — scrolling, gaming and video playback all feel so fluid. The Snapdragon 8s Gen 3 chip handles everything I throw at it, from BGMI at high settings to multitasking with zero lag. Battery easily lasts a full day with heavy use, and 80 W fast charging tops it up quickly. The camera is solid for social media shots in daylight — crisp and detailed. Low-light photos are good, though not flagship level. Overall, performance and display are standout features.",

    "After using this as my daily driver for a week, I’m very happy. OxygenOS 15 is clean and responsive, and I don’t feel any slowdown switching between apps. The battery easily lasts till night even with streaming, browsing, and light gaming; the 80 W charging is genuinely fast. The Plus Key is a nice touch for shortcuts, though I wish it were more customizable. Camera quality is dependable for everyday shots. Only reason I didn’t give 5 ★ is because of occasional heating during heavy gaming. ",

    "Display is crisp and vibrant, and performance feels more than adequate for most users. Battery life is decent, though not as exceptional as some claims — you’ll still need to charge daily. Camera does a competent job in daylight but struggles in low light. The ultrawide lens feels a bit weak for landscape shots. Build quality is nice, but the pill-shaped camera bump isn’t my favourite. Overall a good midrange phone, but not a flagship killer.",

    "I bought this specifically for gaming and multimedia — and it absolutely delivers. The screen’s refresh rate and color depth make games look stunning, and the Snapdragon chipset stays cool even during extended sessions. Speakers are loud and clear for watching movies. The phone doesn’t feel premium in terms of materials as some flagship devices, but everywhere else it outperforms expectations. Battery is huge, and I rarely worry about running out before the day’s end.",

    "Snapdragon 8s Gen 3 sounds great, but real-world use is just “okay”. The UI feels a bit bloated, and I’ve noticed occasional stutters — especially when opening heavy apps. Camera performance is mixed; selfies are fine but ultrawide shots lack detail. Battery life is average. If you just want basic performance and a big screen, it’s fine, but I expected more given the price bracket.",

    "Seriously impressed. The screen quality alone is worth the price — bright, smooth, and vibrant. Gaming performance is excellent, and the combination of RAM and storage means no lag even with dozens of apps open. Battery lasted me all weekend with moderate use. The Plus Key is genuinely something I use daily. Camera quality is solid outside and respectable indoors. Highly recommended for most users.",

    "This phone handles all the basics — browsing, chatting, videos — without hesitation. OxygenOS feels polished and fluid. Battery life is solid, and charging speed is surprisingly quick. Camera is good for social uploads but doesn’t replace a dedicated camera in low light. Build quality is good, though I expected slightly higher brightness in direct sunlight. Still a reliable choice at this price.",

    "Not bad overall, but when compared with other phones in this range, it doesn’t always win. Screen and performance are great, but the camera and fast-charging speed feel like compromises. Software is okay but sometimes pushes features I don’t use. If price drops during a sale, it becomes a much more compelling buy.",

    "I’m coming from an older device, so the difference is night and day. Interface is smooth, battery easily lasts a day and a half with medium usage, and the primary camera captures crisp pictures. The ultrawide is fine for casual landscapes, and the selfie camera performs well even in dim light. Overall this phone handles everything I need it to and more.",

    "OnePlus has nailed the essentials here: performance, battery, and display. The phone feels snappy, the software is clean, and battery life is reliable. I took it on a weekend trip and never worried about running out of charge. Photo quality is good for everyday photos, though not as strong for night shots. Build quality feels solid, and the fingerprint unlock is fast and accurate. A great all-rounder.",

]

import csv
from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class Review(TypedDict):
    key_words: Annotated[str,"Write down the keywords on which the review is based."]
    summary: Annotated[str,"A brief summary of the review."]
    sentiment: Annotated[str,"The sentiment of the review (positive, neutral, negative)."]
    pros: Annotated[str,"List the pros mentioned in the review."]
    cons: Annotated[str,"List the cons mentioned in the review."]
    recommendation_score: Annotated[float,"A score from 1 to 10 indicating the likelihood of recommending this product."]

model = ChatOpenAI(model="gpt-4.1")
structured_output = model.with_structured_output(Review)

results = []

for review_text in oneplus_nord_5_reviews:
    output = structured_output.invoke(review_text)
    results.append(output)

with open("oneplus_nord_5_reviews.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["key_words","summary","sentiment", "pros", "cons", "recommendation_score"]
    )
    writer.writeheader()
    writer.writerows(results)

print("CSV file created successfully.")
