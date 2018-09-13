import csv
import requests
import xml.etree.ElementTree as ET
import os
import contractions

def parseXML(xmlfile):
    tree = ET.parse(xmlfile)

    root = tree.getroot()

    sentences = []

    contract_not = False

    wordcount = 0
    emptycount = 0

    for speaker in root: #each speaker defines a new sentence
        sentence = ""
        sentence+=("<sos> ") #start of sentence token
        wordcount+=1
        for word_element in speaker:
            #print("word element is "+str(word_element))
            word = str(word_element.text)
            temp = ""
            eos = False

            if contract_not:
                if word[-1] in ["?",".","!"]:
                    eos = True
                contract_not = False
                pass
            else:
                #print("the word is "+word)
                word = word.lower()
                if word == 'none':
                    pass
                else:
                    if word[-1] in ["?", ".", "!"]:
                        eos = True

                    if word[-1] == '\'' and len(word) > 2 and word[-2] == "n":
                        contract_not = True

                    for char in word:
                        if char.isalpha():
                            temp+=char
                    if contract_not:
                        sentence+=(temp[:-1])
                        sentence+=(" not ")
                        wordcount+=1

                    elif temp == "il":
                        sentence+="will "
                        wordcount+=1
                    else:
                        if len(temp) > 0:
                            sentence+=(temp+" ")
                            wordcount+=1
                    if eos:
                        sentence += ("<eos>")
                        wordcount+=1
                        sentences.append(sentence)
                        sentence = ""
                        sentence += ("<sos> ")
                        wordcount+=1
        sentence+=("<eos>") #end of sentence token
        wordcount+=1
        if sentence == "<sos> <eos>":
            wordcount -= 2
            emptycount+=1
            pass
        else:
            sentences.append(contractions.fix(sentence))
            wordcount += len(contractions.fix(sentence).split()) - len(sentence.split())
            #we got some new words by removing contractions, so add those to the word count
        #print(sentence)

    return sentences, wordcount, emptycount

def src_to_txts(src_directory, src_extension, txt_directory):
    src_directory_enc = os.fsencode(src_directory)

    wordcount = 0
    wordcounttemp = 0
    for file in os.listdir(src_directory_enc):
        filename = os.fsdecode(file)

        if filename.endswith("."+src_extension):
            target_file = open(txt_directory+filename[:-4]+".txt","w+")


            sentences, wordcounttemp, emptycount = parseXML(src_directory + filename)
            wordcount += wordcounttemp
            for sentence in sentences:
                target_file.write(sentence)
                target_file.write("\n")
            target_file.close()
    print("Txt files written with "+str(wordcount)+" total tokens.")
    print(str(emptycount)+" many lines were ignored.")

