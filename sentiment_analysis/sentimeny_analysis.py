import pandas as pd
import matplotlib.pyplot as plt
import os

# Dataset load
df = pd.read_csv("sentiment_analysis/datasets/Womens Clothing E-Commerce Reviews.csv")

# Missing reviews remove
df = df.dropna(subset=["Review Text"])

# Sentiment create using rating
df["Sentiment"] = df["Rating"].apply(
    lambda x: "Positive" if x >= 4 else "Negative"
)

# Count sentiments
sentiment_counts = df["Sentiment"].value_counts()

# Images folder
os.makedirs("sentiment_analysis/images", exist_ok=True)

# Graph 1: Bar Chart
plt.figure(figsize=(6, 4))
sentiment_counts.plot(kind="bar")
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.savefig("sentiment_analysis/images/Figure_1.png")
plt.show()

# Graph 2: Pie Chart
plt.figure(figsize=(6, 6))
sentiment_counts.plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.title("Positive vs Negative Reviews")
plt.savefig("sentiment_analysis/images/Figure_2.png")
plt.show()

# Graph 3: Rating Distribution
rating_counts = df["Rating"].value_counts().sort_index()

plt.figure(figsize=(8, 5))
plt.plot(rating_counts.index, rating_counts.values, marker="o")
plt.title("Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Number of Reviews")
plt.grid(True)
plt.savefig("sentiment_analysis/images/Figure_3.png")
plt.show()

print("Analysis Completed Successfully!")
