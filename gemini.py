import google.generativeai as genai

genai.configure(api_key="AIzaSyDFUKW4QZ0WeQw5_Bz9kbinynstDL8ayL0")
model = genai.GenerativeModel("gemini-2.0-flash")

# Initial greeting
response = model.generate_content("Xin chào!")
print(response.text)

# Chat loop
while True:
    try:
        user_input = input("User: ").strip()
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        
        # Generate response using Gemini
        response = model.generate_content(user_input)
        print("Assistant:", response.text)
        
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        break