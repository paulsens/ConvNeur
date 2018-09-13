from Socket import openSocket, sendMessage, partMess
from Init import joinRoom
import string
from Read import getUser,getMessage
import datetime
from Settings import CHANNEL
import CNeur_Model


bot = CNeur_Model.CNeural(input_td = "ConvNeur/input_td.txt", output_td = "ConvNeur/output_td.txt",
              input_valid = "ConvNeur/input_valid.txt", output_valid = "ConvNeur/output_valid.txt",
              tdcount = 2509056, validcount = 282304)

bot.twitch_build_inference(weights_file="weights1.h5", load=True)

s = openSocket() #create our socket
joinRoom(s) #join the server with socket s
readbuffer = ""
pongMess = "PONG :tmi.twitch.tv\r\n"
stay = True
filename = CHANNEL+str(datetime.datetime.now())+".txt"
outFile = open(filename, "w")

#####message check tuples
gl_tuple = ("gl", "gl!", "good luck", "good luck!")
ddd_tuple = ("ddd", "DDD", "3D", "3d")
sid_tuple = ("!BigSid", "!bigsid", "sid")

wr_cmd = "!wr"
bot_cmd = "!kbotm"

with open("KBotM_"+str(datetime.datetime.now())+".txt", "w") as botfile:
    while stay == True:
        readbuffer = readbuffer + s.recv(1024).decode()
        temp = readbuffer.split('\n')

        readbuffer = temp.pop()


        for line in temp:
            if ("PING :tmi.twitch.tv" in line):
                s.send(pongMess.encode('utf-8'))
                print("we ponged")
                # elif ("nice" in message):
                # sendMessage(s, "Yeah it is nice.")
            elif ("Double Tap" in line):
                sendMessage(s, "***Disconnecting***")
                s.send(partMess.encode('utf-8'))

                stay = False
                #print("set stay to False")
            else:
                user = getUser(line)
                message = getMessage(line)
                #message = message.split(' ')
                print("the message is "+message+"\n")

                if ((user != "xalbert_the_dusk") and (user != "albert_the_shadow") and (user != "nightbot") and (
                        user != "moobot")):
                    outFile.write(message + "\n")

                    # print(user + " typed :" + message)
                if (message.split(" ")[0] == "KBotM,"):
                    # someone is talking to the bot
                    if "<quit>" in message:
                        sendMessage(s, "Quitting...")
                        s.send(partMess.encode('utf-8'))
                        stay = False
                    else:
                        print("the context is " + (message[7:].lower()) + " check check\n")
                        response = bot.twitch_inference(message[7:].lower())

                        botfile.write(message.lower() + "\n")
                        botfile.write(response + "\n")
                        sendMessage(s, response)

                elif (any(t in message.lower() for t in gl_tuple)):
                    sendMessage(s, "You're supposed to say bad luck on this channel.")
                elif(wr_cmd in message.lower()):
                    sendMessage(s, "World record is your mom, ok?")
                #elif (any(t in message for t in ddd_tuple)):
                    #sendMessage(s, "Shut up, Dream Drop is fun.")
                elif(bot_cmd in message.lower()):
                    response = "I am a neural-net AI written by KBM. You can talk to me by typing KBotM, _____ "
                    sendMessage(s, response)
                    response = "I am not Albert. I don't want keywords, I want an actual sentence or question."
                    sendMessage(s, response)
                    response = "Capitalization does not matter except for my name, but you must punctuate the end of your statement/question."
                    sendMessage(s, response)

outFile.close()




