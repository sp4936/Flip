import openai
import pandas as pd  # For handling the CSV file
from gtts import gTTS
import IPython.display as ipd
import asyncio
import time

# Step 1: Set Up API Keys and Endpoints
openai.api_key = ''

# Step 2: Load CSV File with Product Information
def load_product_data(csv_file):
    """
    Load product data from a CSV file into a DataFrame.
    
    :param csv_file: Path to the CSV file containing product information
    :return: DataFrame with product information
    """
    return pd.read_csv(csv_file)

# Step 3: Fetch Product Information by ID
def fetch_product_by_id(product_data, product_id):
    """
    Fetch product information from the DataFrame based on the given product ID.
    
    :param product_data: DataFrame with product information
    :param product_id: The unique product ID to search for
    :return: Dictionary containing product details
    """
    product_row = product_data[product_data['uniq_id'] == product_id]
    
    if not product_row.empty:
        return {
            "product_name": product_row['product_name'].values[0],
            "description": product_row['description'].values[0],
            "features": product_row['product_specifications'].values[0],
            "brand": product_row['brand'].values[0],
            "price": product_row['retail_price'].values[0],
            "discounted_price": product_row['discounted_price'].values[0]
        }
    else:
        print("Product not found")
        return None

# Step 4: Define Asynchronous Function to Get Product Summary from LLM
async def get_product_summary(product_data):
    """
    Sends the collected product data to the LLM and returns a summary.
    This function is asynchronous to reduce latency.
    """
    prompt = (f"Provide a detailed but concise description for the following product: "
              f"Product Name: {product_data['product_name']}. "
              f"Description: {product_data['description']}. "
              f"Features: {product_data['features']}. "
              f"Brand: {product_data['brand']}. "
              f"Retail Price: {product_data['price']} USD. "
              f"Discounted Price: {product_data['discounted_price']} USD.")
    
    start_time = time.time()

    response = await openai.ChatCompletion.acreate(
        engine="gpt-4o-mini",  # Use the gpt-4o-mini engine
        messages=[
            {"role": "system", "content": "You are a knowledgeable product seller."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,  # Reduce token count for faster response
        n=1,
        stop=None,
        temperature=0.5  # Lower temperature for more predictable output
    )
    
    end_time = time.time()
    latency = end_time - start_time
    print(f"LLM Response Latency: {latency * 1000:.2f} ms")

    summary = response['choices'][0]['message']['content'].strip()
    return summary

# Step 5: Define Function to Convert Text to Speech
def text_to_speech(text, lang='en'):
    """
    Convert the given text to speech using Google Text-to-Speech.
    """
    tts = gTTS(text=text, lang=lang)
    tts.save("response.mp3")
    ipd.display(ipd.Audio("response.mp3"))

# Step 6: Define Function for Real-Time Interaction Using Typed Input
async def real_time_conversation(product_data):
    """
    Handle real-time interaction with the user via typed input.
    """
    while True:
        product_id = input("Please enter the Product ID (or type 'exit' to quit): ")
        
        if product_id.lower() == 'exit':
            print("Exiting the conversation.")
            break

        # Fetch product information based on the typed product ID
        product_info = fetch_product_by_id(product_data, product_id)
        if product_info is None:
            continue

        # Generate product summary or response from LLM
        summary = await get_product_summary(product_info)
        print(f"AI Response: {summary}")

        # Convert response to speech
        text_to_speech(summary)

# Step 7: Main Execution
def main():
    # Load the product data
    csv_file = "flipkart_com-ecommerce_sample.csv"  # Replace with your actual CSV file path
    product_data = load_product_data(csv_file)
    
    # Run the real-time conversation function
    asyncio.run(real_time_conversation(product_data))

if __name__ == "__main__":
    main()
