import requests
import json
import markdown

def generate_questions(city):
    """
    Generate a list of top 5 jokes using Ollama's API.
    """
    prompt = f"Act as a comedy expert at the {city} comedy show. Generate a list of the top 5 jokes that create laughter at {city}."

    response = requests.post('http://localhost:11434/api/chat', 
                             json={
                                 'model': 'llama3.2:3b-instruct-fp16',
                                 'messages': [{"role": "user", "content": prompt}]
                             },
                             stream=True)

    full_response = ""
    for line in response.iter_lines():
        if line:
            try:
                json_response = json.loads(line)
                if 'response' in json_response:
                    full_response += json_response['response']
                if json_response.get('done', False):
                    break
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print(f"Problematic line: {line}")

    return full_response

def display_questions_with_markdown(city):
    """
    Generate and display the top 5 jokes.
    """
    questions_text = generate_questions(city)

    markdown_output = f"### Top 5 jokes at {city}:\n\n{questions_text}"

    html_output = markdown.markdown(markdown_output)

    print(html_output)

    print("\nRaw Markdown:")
    print(markdown_output)
