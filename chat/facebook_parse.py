# Read, clean and store messages from facebook to json format

from bs4 import BeautifulSoup
import json

html_doc = 'messages.htm'
html_text = open(html_doc, 'r', encoding="utf8").read()

print("\n...Parsing html\n")
soup = BeautifulSoup(html_text, "html.parser")
print("\n...HTML parsed\n")

# Dict with all conversations, turned into json
chats = {}

# Get list of conversations
threads = soup.find_all("div", class_="thread")
print("\n...threads found\n")

# Index for each chat
idx = 0

# For each conversation
for thread in threads[:]:

    print("\nProcessing chat {0}\n".format(idx))

    # Get list of messages (index 0 is just an id, not usefull)
    messages = str(thread).split('<div class="message"')[1:]
    # Messages are stored from last to first, so list is reversed
    messages = messages[::-1]

    # List to save messages, each message appended as a dict
    chat = []

    # Timestamp to identify messages in time
    timestamp = 0

    # Get author name and message
    for message in messages:
        message_soup = BeautifulSoup(message, "html.parser")

        # Get name
        name = message_soup.span.get_text()
        # Get message
        text = message_soup.p.get_text()

        # Append message to chat list as a dict
        chat.append({'author': name, 'text': text, 'timestamp': timestamp})
        timestamp += 1

    # Append list of messages to dict of all chats
    chats[idx] = chat
    print("\n...Chat " + str(idx) + " parsed\n")
    idx += 1

# Save chats to json
with open('chats.json', 'a', encoding="utf-8") as file:
    json.dump(chats, file, ensure_ascii=False)

# with open('chats.json', 'r', encoding="utf-8") as file:
    # data = json.load(file)
