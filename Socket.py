from Settings import *
import socket

partMess = "PART #"+ CHANNEL + "\r\n"

def openSocket():

    s = socket.socket()
    s.connect((HOST,PORT))

    passMess = "PASS " + PASS + "\r\n"
    s.send(passMess.encode('utf-8'))

    nickMess = "NICK " + NICK + "\r\n"
    s.send(nickMess.encode('utf-8'))

    joinMess = "JOIN #" + CHANNEL + "\r\n"
    s.send(joinMess.encode('utf-8'))

    return s


def sendMessage(s,message):
    messageTemp = "PRIVMSG #" + CHANNEL + " :" + message +"\r\n" #PRIVMSG #channel_name :message
    s.send(messageTemp.encode('utf-8'))
    print("Sent: " + messageTemp)
