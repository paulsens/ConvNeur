import string
from Socket import sendMessage

def joinRoom(s):
    readbuffer=""
    Loading=True
    succMess = "I am a bot written by KBM. Blehhhhhhh"

    while Loading:
        readbuffer = readbuffer + s.recv(1024).decode()
        temp = readbuffer.split('\n')

        readbuffer=temp.pop()

        for line in temp:
            print(line)
            Loading = loadingComplete(line)
    #sendMessage(s, succMess) #i wrote sendMessage, it encodes strings, so no need to encode here

def loadingComplete(line):

    if("End of /NAMES list" in line):
        return False
    else:
        return True