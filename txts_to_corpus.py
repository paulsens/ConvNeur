import os
import operator
from random import randint
from CNeur_Constants import LEN_LIMIT
def concat_txts(txt_directory, target_name):
    directory = os.fsencode(txt_directory)
    finalfile = open(target_name+".txt","w+")

    count = 0

    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".txt"):
            with open(txt_directory+filename) as source_file:
                for line in source_file:
                    finalfile.write(line)
                    #finalfile.write("\n")
                    count+=1

    print("Wrote "+str(count)+" lines to file.")

def write_indices_files(src, dict, target):
    with open(src, "r") as fp:
        input = open("input_"+target, "w+")
        output = open("output_"+target, "w+")



        curr_sent = ""

        first_sent = ""
        is_first = True
        sos_idx = dict["<sos>"]

        line = fp.readline()
        while line:
            sentence = line.strip().split()

            curr_sent = ""
            for word in sentence:
                if word not in dict:
                    input.write("0 ")
                    curr_sent += "0 "

                    if is_first:
                        first_sent += "0 "


                else:

                    input.write(str(dict[word]) + " ")
                    curr_sent += str(dict[word]) + " "

                    if is_first:
                        first_sent += str(dict[word]) + " "

            input.write("\n")
            curr_sent += "\n"
            line = fp.readline()



            if not is_first:
                output.write(curr_sent)
                #print("wrote to output "+curr_sent)


            if not line: #write first sentence as target of last sentence
                output.write(first_sent+"\n")
                #print("wrote to output "+first_sent+"\n")
            is_first = False

        input.close()
        output.close()


        return "input_"+target, "output_"+target

def sort(input, output, batch_size):


    #group up the training data in sets of batch_size so that the average length within the group is minimized
    #we want each batch to have small variance in sentence length
    #create numpy arrays padded with zeros so that each batch has the same length
    #however, in each batch, input and output need not be the same length.
    #reverse the order of the input sequence
    #i.e the problem of mapping [a, b, c] -> [d, e]
                        #and    [f, g] -> [h, i, j]
                        #becomes [c, b, a] -> [d, e, 0]
                        #and     [0, g, f] -> [h, i, j]

    mapping = []
    idx = 0
    with open(input, "r") as ifp, open(output, "r") as ofp:
        iline = ifp.readline()
        oline = ofp.readline()
        o_offset = 0
        i_offset = 0
        next_i_offset = 0
        next_o_offset = 0

        while iline:
            o_offset = next_o_offset
            i_offset = next_i_offset
            #we need the offsets calculated at the end of the last line


            next_i_offset += len(iline)
            next_o_offset += len(oline)
            iline = iline.split()
            oline = oline.split()


            values = [idx, len(iline), i_offset, len(oline), o_offset]
            #print("values is "+str(values))
            #so this is the line number, the lengths of the lines, and their offsets, which will
            #let us find the lines without having to read in the entire file

            iline=ifp.readline()
            oline=ofp.readline()
            idx+=1
            mapping.append(values)

        #print("before sorting mapping is " + str(mapping))
        mapping = sorted(mapping, key=operator.itemgetter(3))
        #print("after sorting, mapping is "+str(mapping))

    #now we actually sort
        with open("input_indices_sorted.txt", "w+") as isfp, open("output_indices_sorted.txt","w+") as osfp:
            for i in range(0, idx):
                i_off = mapping[i][2]
                o_off = mapping[i][4]

                ifp.seek(i_off)
                i_line = ifp.readline().strip()
                isfp.write(i_line+"\n")

                ofp.seek(o_off)
                o_line = ofp.readline().strip()
                osfp.write(o_line+"\n")

    return "input_indices_sorted.txt", "output_indices_sorted.txt"






def pad(myfile, bs): #filepath and batch size
    with open(myfile, "r") as fp, open(myfile[:-4]+"_padded.txt", "w+") as outfile:
        line = fp.readline()
        count = 0
        lines = []
        max = 0
        while line:
            line = line.strip()
            if len(line.split()) > max:
                max = len(line.split())
            lines.append(line.split())
            count+=1

            if(count == bs): #we've finished collecting a batch and have the max length
                for i in range(0, bs):
                    diff = max - len(lines[i])
                    for j in range(0, diff):
                        lines[i].append("0")
                #each line should be padded now
                    outfile.write(" ".join(lines[i])+"\n") #write the padded line to file
                count = 0
                max = 0
                lines = []
            line = fp.readline()
        #add any remaining lines
        for i in range(0, len(lines)):
            diff = max - len(lines[i])
            for j in range(0, diff):
                lines[i].append("0")
            # each line should be padded now

            outfile.write(" ".join(lines[i]) + "\n")  # write the padded line to file

    return myfile[:-4]+"_padded.txt"

#1506.05869 sees improved results by reversing the input strings
def reverse(myfile):
    with open(myfile, "r") as fp, open(myfile[:5]+"_reversed.txt", "w+") as outfile:
        line = fp.readline()
        while line:
            words = (line.strip()).split()
            words = list(reversed(words))

            outfile.write(" ".join(words)+"\n")
            line = fp.readline()

    return myfile[:5]+"_reversed.txt"

def make_input_val(val_pct, input_file, bs):
    #val_pct is a 2 digit number, if our randint is >= to it then that line goes to the validation set
    tdcount = 0
    validcount = 0


    with open(input_file, "r") as fp, open("input_td.txt","w+") as tdfile, open("input_valid.txt","w+") as valfile:

        b_count = 0 #say the first batch in the file is the 0th batch
        validbatches = []
        line = fp.readline()
        while line:
            choice = randint(1, 100)
            line = line.strip()
            if choice <= val_pct:
                #the files are sorted into batches of similar length, so let's write an entire batch at a time
                for i in range(0, bs):

                    valfile.write(line+"\n")
                    validcount += 1
                    line = fp.readline()
                    if not line:
                        break
                    line = line.strip()

                validbatches.append(b_count)
                b_count+=1

            else:
                for i in range(0, bs):

                    tdfile.write(line + "\n")
                    tdcount += 1
                    line = fp.readline()
                    if not line:
                        break
                    line = line.strip()

                b_count += 1
    print("There are "+str(tdcount)+" lines for training, and "+str(validcount)+" lines for validation.")

    return validbatches, "input_td.txt", "input_valid.txt", tdcount, validcount
    #return our list of batches that were sent to validation, so that the output can be matched up

def make_output_val(output_file, bs, batchlist):
    batchcount = 0
    idx = 0

    with open(output_file, "r") as fp, open("output_td.txt","w+") as tdfile, open("output_valid.txt","w+") as valfile:
        line = fp.readline()
        while line:
            line = line.strip()

            if idx < len(batchlist) and batchlist[idx] == batchcount:
                for i in range(0, bs):
                    valfile.write(line+"\n")
                    line = fp.readline()
                    if not line:
                        break
                    line = line.strip()
                batchcount+=1
                idx+=1
                #we've finished a batch and we want to check the next entry in batchlist (it's sorted, so this is fine)
            else: #this batch is not for the validation set
                for i in range(0, bs):
                    tdfile.write(line+"\n")
                    line = fp.readline()
                    if not line:
                        break
                    line = line.strip()
                batchcount+=1

    return "output_td.txt", "output_valid.txt"

def scrub(input_file):
    emptycount = 0

    with open(input_file, "r") as ifp, open(input_file[:-12]+"_scrubbed.txt", "w+") as ofp:
        line = ifp.readline()
        not_empty = False
        while line:
            line = line.strip().split()
            if line != ["1","2"]:
                ofp.write(" ".join(line))
                not_empty = True
            else:
                emptycount +=1
            line = ifp.readline()
            if line:
                if not_empty:
                    ofp.write("\n")
                    not_empty = False
    print("Scrubbed "+str(emptycount)+" empty lines from file.")

    return input_file[:-12]+"_scrubbed.txt"

def remove_bad_batches(inputfile, outputfile, batch_size):
    with open(inputfile, "r") as ifp, open("input_batches.txt", "w+") as irfp, open(outputfile,"r") as ofp, open("output_batches.txt","w+") as orfp:
        inline = ifp.readline()
        outline = ofp.readline()
        skip_batch = False
        zerocount = 0
        inbatch = []
        outbatch = []
        skipcount = 0


        while inline and outline:
            for i in range(0, batch_size):
                inbatch.append(inline.strip())
                outbatch.append(outline.strip())

                outlist = outline.strip().split()

                for j in range(0, len(outlist)):
                    if outlist[j] == "0":
                        zerocount+=1
                if zerocount/len(outlist)>1/3 or len(outlist) > LEN_LIMIT:
                    skip_batch = True
                    #so if any of the lines are more than a third zeroes or is too long, we're skipping the whole batch
                zerocount = 0

                inline = ifp.readline()
                outline = ofp.readline()

                if not inline or not outline:
                    break

            if not skip_batch:
                for i in range(0, len(inbatch)-1):
                    irfp.write(inbatch[i]+"\n")
                    orfp.write(outbatch[i]+"\n")
                irfp.write(inbatch[-1])
                orfp.write(outbatch[-1])
                if inline and outline:
                    irfp.write("\n")
                    orfp.write("\n")

            else:
                skipcount+=1


            skip_batch = False
            inbatch = []
            outbatch = []

        print("We skipped "+str(skipcount)+" batches because they were too heavily zero-padded.")

    return "input_batches.txt", "output_batches.txt"
