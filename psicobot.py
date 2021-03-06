import os
import sys
import json

import requests
from flask import Flask, request

app = Flask(__name__)

PAT = 'EAAGLRiMjkrsBABVa39Mm78ABsMuEKIkZCqq4cvItXZBNL2MtQJTJZBNOPLbDr8rypuf9xuOT0b8eGXbUIKgXtYr4Y6GEtR59Quqr3AzbI36rTKjqZCGXuQDJsAKZAvMyGPxsx1vd7IZA4vOt6BXkpGvugh9AtUi3E1bFSe6UJO3gZDZD'

@app.route('/', methods=['GET'])
def verify():
    # when the endpoint is registered as a webhook, it must echo back
    # the 'hub.challenge' value it receives in the query arguments
    if request.args.get("hub.mode") == "subscribe" and request.args.get("hub.challenge"):
        if not request.args.get('hub.verify_token', '') == 'my_voice_is_my_password_verify_me':
            return "Verification token mismatch", 403
        return request.args.get('hub.challenge', '')

    return "Hello world", 200


@app.route('/', methods=['POST'])
def webhook():

    # endpoint for processing incoming messaging events

    data = request.get_json()
    log(data)  # you may not want to log every incoming message in production, but it's good for testing

    if data["object"] == "page":

        for entry in data["entry"]:
            for messaging_event in entry["messaging"]:

                if messaging_event.get("message"):  # someone sent us a message

                    sender_id = messaging_event["sender"]["id"]        # the facebook ID of the person sending you the message
                    recipient_id = messaging_event["recipient"]["id"]  # the recipient's ID, which should be your page's facebook ID
                    message_text = messaging_event["message"]["text"]  # the message's text

                    decideMessage(sender_id, message_text)
                    #send_message(sender_id, "got it, thanks!")

                if messaging_event.get("delivery"):  # delivery confirmation
                    pass

                if messaging_event.get("optin"):  # optin confirmation
                    pass

                if messaging_event.get("postback"):  # user clicked/tapped "postback" button in earlier message
                  sender_id = messaging_event["sender"]["id"]
                  message_text = messaging_event["postback"]["payload"]

                  message_text = message_text.lower()

                  if message_text == "si":
                    send_message(sender_id, "Por favor contesta las siguientes preguntas")
                  elif message_text == "no":
                    send_message(sender_id, "Si cambias de opinion contactanos de nuevo :)")

    return "ok", 200

def decideMessage(sender_id, message_text):
  text = message_text.lower()
  if "hola" in text:
    sendButtonMessage(sender_id)
  elif "adios" in text:
    send_message(sender_id, "Hasta luego!")
  else:
    send_message(sender_id, "Visto")

def sendButtonMessage(sender_id):
  message_data = {
    "attachment":{
      "type":"template",
      "payload":{
        "template_type":"button",
        "text":"Te gustaria recibir atencion de un psicologo?",
        "buttons":[
          {
            "type":"postback",
            "title":"Si",
            "payload":"si"
          },
          {
            "type":"postback",
            "title":"No",
            "payload":"no"
          }
        ]
      }
    }
  }

  sendRequest(sender_id, message_data)

def send_message(recipient_id, message_text):

    log("sending message to {recipient}: {text}".format(recipient=recipient_id, text=message_text))

    sendRequest(recipient_id, {"text":message_text})

def sendRequest(recipient_id, message_data):
    params = {
        "access_token": PAT
    }
    headers = {
        "Content-Type": "application/json"
    }
    data = json.dumps({
        "recipient": {
            "id": recipient_id
        },
        "message": message_data
    })
    r = requests.post("https://graph.facebook.com/v2.6/me/messages", params=params, headers=headers, data=data)
    if r.status_code != 200:
        log("Eror :c")
        log(r.status_code)
        log(r.text)

def log(message):  # simple wrapper for logging to stdout on heroku
    print str(message)
    sys.stdout.flush()


if __name__ == '__main__':
    app.run(debug=True)