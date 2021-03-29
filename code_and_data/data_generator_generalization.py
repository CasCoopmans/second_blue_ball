import numpy as np
import pandas as pd
import pdb
import random

##################################################
#### generate generalization data (interpolation & 
#### extrapolation using one-hot vectors as input
##################################################

def gen_hier_onehot(train_size = 2, gen_type = 'int', random_training = False):
    
    target = []
    input_batch = []
    output_batch = []
    
    # words in one-hot encoding
    seventh =  [1,0,0,0,0,0,0,0,0,0]
    second =   [0,1,0,0,0,0,0,0,0,0]
    third =    [0,0,1,0,0,0,0,0,0,0]
    fourth =   [0,0,0,1,0,0,0,0,0,0]
    fifth =    [0,0,0,0,1,0,0,0,0,0]
    sixth =    [0,0,0,0,0,1,0,0,0,0]
    blue =     [0,0,0,0,0,0,1,0,0,0]
    green =    [0,0,0,0,0,0,0,1,0,0]
    red =      [0,0,0,0,0,0,0,0,1,0]
    ball =     [0,0,0,0,0,0,0,0,0,1]

    # combines words into their category 
    ordinals = [second, third, fourth, fifth, sixth, seventh] # 'first' not included because all trials must be divergent
    colors = [blue, red, green]
    colors_nored = [blue, green]
    shapes = [ball]
    
    # properties of elements in picture
    blue_p   = [1,0,0,0]
    green_p  = [0,1,0,0]
    red_p    = [0,0,1,0]
    ball_p   = [0,0,0,1]
    
    # combine properties of elements in picture
    colors_p = [blue_p, red_p]
    colors_p_nored = [blue_p, green_p]
    shapes_p = [ball_p]

    # define number of target-present and target-absent trials
    targetpresencetrain = ['present','absent'] * int((train_size)/2) 
    targetpresencetest = ['present','absent'] * int((100)/2)     
    random.shuffle(targetpresencetrain) 
    random.shuffle(targetpresencetest) 
   
    # define possible target positions for pseudorandom simulation 
    possibletargpos = [4,5,6,7,8,9] 
    
    # training size is input to the function; add 100 test trials
    runs = train_size + 100

    # generate training and test trials
    for runs in range(0,runs):
            
            if runs < train_size:
                trial = targetpresencetrain[runs]
            else:
                trial = targetpresencetest[runs-train_size]
            
            # run the loop until an input-output combination is generated that is new
            is_trial_new = False
            while is_trial_new == False:

                ### generate phrase
                
                # pick random properties for target phrase
                ordinal = random.choice(ordinals)
                shape = random.choice(shapes)

                # the properties of possible targets depend on type of generalization test
                if gen_type == 'ext':
                    if runs < train_size:
                        color = random.choice(colors_nored)
                    else:
                        ordinal = [0,0,1,0,0,0,0,0,0,0] # ordinal is third
                        color = [0,0,0,0,0,0,0,0,1,0]  # color is red
                elif gen_type == 'int':
                    if runs < train_size:
                        color = random.choice(colors)
                        if ordinal == [0,0,1,0,0,0,0,0,0,0]: # if ordinal is third
                            color = random.choice(colors_nored) # then color cannot be red
                    else:
                        color = [0,0,0,0,0,0,0,0,1,0]  # color is red
                        ordinal = [0,0,1,0,0,0,0,0,0,0] # ordinal is third
                
                # the target item for in the picture
                item_p = [[],[]]
                
                # save properties of target for creating picture
                if (color == blue) == True:
                    item_p[0] = [1,0,0,0]
                if (color == green) == True:
                    item_p[0] = [0,1,0,0]
                if (color == red) == True:
                    item_p[0] = [0,0,1,0]
                if (shape == ball) == True:
                    item_p[1] = [0,0,0,1]

                # get what the ordinal was
                for ord in range(9):
                    if ordinal[ord] == 1:
                        ordnum = ord + 1
                        if ordnum == 1: 
                            ordnum = 7 # seventh replaces first

                ### generate a picture
               
                # run the loop to make sure the trial is divergent: 
                # the linear interpretation is not the hierarchical interpretation (if linear target is present)
                is_picture_divergent = False
                while is_picture_divergent == False:

                    # the properties of possible items in picture depend on type of generalization test
                    if gen_type == 'ext':
                        if runs < train_size:
                            colors_p = [blue_p, green_p]
                        else:
                            colors_p = [blue_p, green_p, red_p]
                    elif gen_type == 'int':
                        colors_p = [blue_p, green_p, red_p]
                    
                    # define all elements in the picture
                    picture = []
                    for i in range(8): # picture has len = 8
                        picture.append([random.choice(colors_p),random.choice(shapes_p)])   
                    
                    # uncomment if linear answer must be in the picture
                    #picture[ordnum-1] = item_p

                    # count how many items are the same as the target
                    count = 0
                    for elements in range(len(picture)):
                        if picture[elements] == item_p:
                            count += 1 

                    # make sure trial is divergent on target-present trials:
                    # run through picture and count how many elements to the left of the target are the same as the target 
                    targetnum = 0
                    for elements in range(ordnum-1):
                        if picture[elements] == item_p:
                            targetnum += 1
                    
                    if trial == 'present':
                        # if there are enough items the same as the target (so the hierarchical target is present)
                        if count >= (ordnum):
                            # and if the trial is divergent
                            if targetnum != ordnum-1:
                                is_picture_divergent = True
                    elif trial == 'absent':
                        if count < (ordnum):
                            is_picture_divergent = True                            
                
                if trial == 'present':
                    # now that we know that the hierarchical target is present, find it
                    targetfound = False
                    while targetfound == False:
                        targetnum = 0 
                        # run through all elements in the picture
                        for elements in range(len(picture)):
                            # if an element is the same as the target, count it
                            if picture[elements] == item_p:
                                targetnum += 1
                                # if the ordinal of target is reached, save target position and exit the loop
                                if targetnum == ordnum:
                                    targetposition = elements+1
                                    targetfound = True
                elif trial == 'absent':
                    targetposition = 9
               
                # to generate pseudorandom input-output mappings
                if random_training == True:
                    targetposition = random.choice(possibletargpos)
                
                # save the output vector
                output_vec = [0] * 9
                output_vec[targetposition-1] = 1
                                            
                # flatten the picture, which is an array of arrays
                pict_flat = [item for sublist in picture for item in sublist]
                pict_flat = [item for sublist in pict_flat for item in sublist]
                pict_flat = np.array(pict_flat)

                # normalize the picture to make sure it has "length" 1 (length defined as Euclidean distance), as all one-hot vectors do
                # this ensures that the input to the lstm has the same "net content" across all vectors
                pict_norm = pict_flat / np.sqrt(np.sum(pict_flat**2))
                
                # add to the left of the picture 10 zeros to incorporate the length of each word of the phrase 
                pict_pad = np.pad(pict_norm,(10,0), 'constant')
                
                # add to the right of each word 64 zeros to incorporate the length of the picture
                ordinal_pad = np.pad(ordinal,(0,64), 'constant')
                color_pad = np.pad(color,(0,64), 'constant')
                shape_pad = np.pad(shape,(0,64), 'constant')
                
                # combine all three words and the picture into one flat list
                input_vec = [list(ordinal_pad),list(color_pad),list(shape_pad),list(pict_pad)]
                input_vec = [s for sublist in input_vec for s in sublist]
                
                # cut the list up into four equal-sized numpy arrays
                input_vec = np.array(input_vec)
                input_vec = np.reshape(input_vec,newshape=(4,74))
          
                # check whether the input-output combination is new:
                # compare it to all previously generated examples in input_batch and exit loop if the generated trial is new
                if runs == 0:
                    is_trial_new = True
                else:
                    ## make sure that the input picture is new/hasn't been created before
                    newness = np.zeros(runs)
                    for inputs in range(1,runs):
                        # element-wise comparison; if all are True, the two input_vecs were identical
                        if (input_vec == input_batch[inputs-1]).all() == True:
                            newness[inputs] = 1
                            break # stop looping through more runs
                    if sum(newness) == 0.0: # if current input_vec is not identical to any of the previous ones, exit the loop
                        is_trial_new = True
            input_batch.append(input_vec)
            output_batch.append(output_vec)

    # return input and corresponding output for training and set sets                  
    return np.array(input_batch[0:train_size]), np.array(output_batch[0:train_size]), np.array(input_batch[train_size:]), np.array(output_batch[train_size:])


######################################################
#### generate generalization data (interpolation & 
#### extrapolation using full word embeddings as input
######################################################

def gen_hier_embful(train_size = 2, gen_type = 'int'):
 
    target = []
    input_batch = []
    output_batch = []

    # load full word embeddings
    myembeddings = pd.read_csv('word2vec/embeddings.csv', header = 0)
    myembeddings = myembeddings['embedding']
    # convert the embeddings from list to string of floats
    second = np.array(np.matrix(myembeddings[0])).ravel()
    third = np.array(np.matrix(myembeddings[1])).ravel()
    fourth = np.array(np.matrix(myembeddings[2])).ravel()
    fifth = np.array(np.matrix(myembeddings[3])).ravel()
    sixth = np.array(np.matrix(myembeddings[4])).ravel()
    seventh = np.array(np.matrix(myembeddings[5])).ravel()
    blue = np.array(np.matrix(myembeddings[6])).ravel()
    green = np.array(np.matrix(myembeddings[7])).ravel()
    red = np.array(np.matrix(myembeddings[8])).ravel()
    ball = np.array(np.matrix(myembeddings[9])).ravel()
 
    # combines words into their category 
    ordinals = [second, third, fourth, fifth, sixth, seventh] # 'first' not included because all trials must be divergent
    colors = [blue, green, red]
    colors_nored = [blue, green]
    shapes = [ball]
    
    # words in one-hot encoding; for the picture
    blue_p =     [1,0,0,0]
    green_p =    [0,1,0,0]
    red_p =      [0,0,1,0]
    ball_p =     [0,0,0,1]
    
    # combine properties of elements in picture
    colors_p = [blue_p, green_p, red_p]
    colors_p_nored = [blue_p, green_p]
    shapes_p = [ball_p]
     
    # define number of target-present and target-absent trials
    targetpresencetrain = ['present', 'absent'] * int((train_size)/2) 
    targetpresencetest = ['present', 'absent'] * int((100)/2)     
    random.shuffle(targetpresencetrain) 
    random.shuffle(targetpresencetest) 

    # training size is input to the function; add 100 test trials
    runs = train_size + 100    
    
    # generate training and test trials
    for runs in range(0,runs):
            
            if runs < train_size:
                trial = targetpresencetrain[runs]
            else:
                trial = targetpresencetest[runs-train_size]
            
            # run the loop until an input-output combination is generated that is new
            is_trial_new = False
            while is_trial_new == False:

                ### generate phrase
                
                # pick random properties for target phrase
                ordinal = random.choice(ordinals)
                shape = random.choice(shapes)

                # the properties of possible targets depend on type of generalization test
                if gen_type == 'int':
                    if runs < train_size:
                        color = random.choice(colors_nored)
                    else:
                        color = red  # color is red
                        ordinal = third # ordinal is third
                elif gen_type == 'ext':
                    if runs < train_size:
                        color = random.choice(colors)
                        if (ordinal == third).all() == True: # if ordinal is third
                            color = random.choice(colors_nored) # then color cannot be red
                    else:
                        color = red  # color is red
                        ordinal = third # ordinal is third
                
                # the target item for in the picture
                item_p = [[],[]]
                
                # save properties of target for creating picture
                if (color == blue).all() == True:
                    item_p[0] = [1,0,0,0]
                elif (color == green).all() == True:
                    item_p[0] = [0,1,0,0]
                elif (color == red).all() == True:
                    item_p[0] = [0,0,1,0]
                if (shape == ball).all() == True:
                    item_p[1] = [0,0,0,1]

                # save what the ordinal was
                if (ordinal == second).all() == True:
                    ordnum = 2
                elif (ordinal == third).all() == True:
                    ordnum = 3
                elif (ordinal == fourth).all() == True:
                    ordnum = 4
                elif (ordinal == fifth).all() == True:
                    ordnum = 5
                elif (ordinal == sixth).all() == True:
                    ordnum = 6
                elif (ordinal == seventh).all() == True:
                    ordnum = 7
                
                ### generate a picture
                
                # run the loop to make sure the hierarchical target is there
                # and the trial is divergent
                is_picture_divergent = False
                while is_picture_divergent == False:
                    count = 0

                    # the properties of possible items in picture depend on type of generalization test
                    if gen_type == 'int':
                        if runs < train_size:
                            colors_p = [blue_p, green_p]
                        else:
                            colors_p = [blue_p, green_p, red_p]
                    elif gen_type == 'ext':
                        colors_p = [blue_p, green_p, red_p]
                    
                    # define all elements in the picture
                    picture = []
                    for i in range(8): # picture has len = 8
                        picture.append([random.choice(colors_p),random.choice(shapes_p)])                    
                    
                    # uncomment if linear answer must be in the picture
                    #picture[ordnum-1] = item_p
                    
                    # count how many items are the same as the target
                    count = 0
                    for elements in range(len(picture)):
                        if picture[elements] == item_p:
                            count += 1 

                    # make sure trial is divergent on target-present trials:
                    # run through picture and count how many elements to the left of the target are the same as the target 
                    targetnum = 0
                    for elements in range(ordnum-1):
                        if picture[elements] == item_p:
                            targetnum += 1
                    
                    if trial == 'present':
                        # if there are enough items the same as the target (so the hierarchical target is present)
                        if count >= (ordnum):
                            # and if the trial is divergent
                            if targetnum != ordnum-1:
                                is_picture_divergent = True
                    elif trial == 'absent':
                        if count < (ordnum):
                            is_picture_divergent = True                            
                
                if trial == 'present':
                    # now that we know that the hierarchical target is present, find it
                    targetfound = False
                    while targetfound == False:
                        targetnum = 0 
                        # run through all elements in the picture
                        for elements in range(len(picture)):
                            # if an element is the same as the target, count it
                            if picture[elements] == item_p:
                                targetnum += 1
                                # if the ordinal of target is reached, save target position and exit the loop
                                if targetnum == ordnum:
                                    targetposition = elements+1
                                    targetfound = True
                elif trial == 'absent':
                    targetposition = 9                
                
                # save the output vector
                output_vec = [0] * 9
                output_vec[targetposition-1] = 1
                                            
                # flatten the picture, which is an array of arrays
                pict_flat = [item for sublist in picture for item in sublist]
                pict_flat = [item for sublist in pict_flat for item in sublist]
                pict_flat = np.array(pict_flat)

                # normalize the picture to make sure it has "length" 1 (length defined as Euclidean distance), as all one-hot vectors do
                # this ensures that the input to the lstm has the same "net content" across all vectors
                pict_norm = pict_flat / np.sqrt(np.sum(pict_flat**2))
                
                # add to the left of the picture 300 zeros to incorporate the length of each full embedding 
                pict_pad = np.pad(pict_norm,(300,0), 'constant')
                
                # add to the right of each word 64 zeros to incorporate the length of the picture
                ordinal_pad = np.pad(ordinal,(0,64), 'constant')
                color_pad = np.pad(color,(0,64), 'constant')
                shape_pad = np.pad(shape,(0,64), 'constant')
                
                # combine all three words and the picture into one flat list
                input_vec = [list(ordinal_pad),list(color_pad),list(shape_pad),list(pict_pad)]
                input_vec = [s for sublist in input_vec for s in sublist]
                
                # cut the list up into four equal-sized numpy arrays
                input_vec = np.array(input_vec)
                input_vec = np.reshape(input_vec,newshape=(4,364))
          
                # check whether the input-output combination is new:
                # compare it to all previously generated examples in input_batch and exit loop if the generated trial is new
                if runs == 0:
                    is_trial_new = True
                else:
                    ## make sure that the input picture is new/hasn't been created before
                    newness = np.zeros(runs)
                    for inputs in range(1,runs):
                        # element-wise comparison; if all are True, the two input_vecs were identical
                        if (input_vec == input_batch[inputs-1]).all() == True:
                            newness[inputs] = 1
                            break # stop looping through more runs
                            
                    if sum(newness) == 0.0: # if current input_vec is not identical to any of the previous ones, exit the loop
                        is_trial_new = True
                        
            input_batch.append(input_vec)
            output_batch.append(output_vec)

    # return input and corresponding output for training and set sets                  
    return np.array(input_batch[0:train_size]), np.array(output_batch[0:train_size]), np.array(input_batch[train_size:]), np.array(output_batch[train_size:])    

##########################################################
#### generate generalization data (interpolation & 
#### extrapolation using reduced word embeddings as input
##########################################################

def gen_hier_embred(train_size = 2, gen_type = 'int'):
 
    target = []
    input_batch = []
    output_batch = []
    
    # load dimensionality-reduced word embeddings
    myembeddings = pd.read_csv('word2vec/reduced_embeddings.csv', header = 0)
    myembeddings = myembeddings['embedding']
    # convert the embeddings from list to string of floats
    second = np.array(np.matrix(myembeddings[0])).ravel()
    third = np.array(np.matrix(myembeddings[1])).ravel()
    fourth = np.array(np.matrix(myembeddings[2])).ravel()
    fifth = np.array(np.matrix(myembeddings[3])).ravel()
    sixth = np.array(np.matrix(myembeddings[4])).ravel()
    seventh = np.array(np.matrix(myembeddings[5])).ravel()
    blue = np.array(np.matrix(myembeddings[6])).ravel()
    green = np.array(np.matrix(myembeddings[7])).ravel()
    red = np.array(np.matrix(myembeddings[8])).ravel()
    ball = np.array(np.matrix(myembeddings[9])).ravel()

    # combines words into their category 
    ordinals = [second, third, fourth, fifth, sixth, seventh] # 'first' not included because all trials must be divergent
    colors = [blue, green, red]
    colors_nored = [blue, green]
    shapes = [ball]
    
    # words in one-hot encoding; for the picture
    blue_p =     [1,0,0,0]
    green_p =    [0,1,0,0]
    red_p =      [0,0,1,0]
    ball_p =     [0,0,0,1]
    
    # combine properties of elements in picture
    colors_p = [blue_p, green_p, red_p]
    colors_p_nored = [blue_p, green_p]
    shapes_p = [ball_p]
    
    # define number of target-present and target-absent trials
    targetpresencetrain = ['present', 'absent'] * int((train_size)/2) 
    targetpresencetest = ['present', 'absent'] * int((100)/2)     
    random.shuffle(targetpresencetrain) 
    random.shuffle(targetpresencetest) 

    # training size is input to the function; add 100 test trials
    runs = train_size + 100    
    
    # generate training and test trials
    for runs in range(0,runs):
            
            if runs < train_size:
                trial = targetpresencetrain[runs]
            else:
                trial = targetpresencetest[runs-train_size]
            
            # run the loop until an input-output combination is generated that is new
            is_trial_new = False
            while is_trial_new == False:

                ### generate phrase
                
                # pick random properties for target phrase
                ordinal = random.choice(ordinals)
                shape = random.choice(shapes)

                # the properties of possible targets depend on type of generalization test
                if gen_type == 'ext':
                    if runs < train_size:
                        color = random.choice(colors_nored)
                    else:
                        ordinal = third # ordinal is third
                        color = red  # color is red
                elif gen_type == 'int':         
                    if runs < train_size:
                        color = random.choice(colors)
                        if (ordinal == third).all() == True: # if ordinal is third
                            color = random.choice(colors_nored) # then color cannot be red
                    else:
                        color = red  # color is red
                        ordinal = third # ordinal is third
                
                # the target item for in the picture
                item_p = [[],[]]
                
                # save properties of target for creating picture
                if (color == blue).all() == True:
                    item_p[0] = [1,0,0,0]
                elif (color == green).all() == True:
                    item_p[0] = [0,1,0,0]
                elif (color == red).all() == True:
                    item_p[0] = [0,0,1,0]
                if (shape == ball).all() == True:
                    item_p[1] = [0,0,0,1]

                # save what the ordinal was
                if (ordinal == second).all() == True:
                    ordnum = 2
                elif (ordinal == third).all() == True:
                    ordnum = 3
                elif (ordinal == fourth).all() == True:
                    ordnum = 4
                elif (ordinal == fifth).all() == True:
                    ordnum = 5
                elif (ordinal == sixth).all() == True:
                    ordnum = 6
                elif (ordinal == seventh).all() == True:
                    ordnum = 7
                
                ### generate a picture
                
                # run the loop to make sure the hierarchical target is there
                # and the trial is divergent
                is_picture_divergent = False
                while is_picture_divergent == False:
                    count = 0

                    # the properties of possible items in picture depend on type of generalization test                    
                    if gen_type == 'ext':
                        if runs < train_size:
                            colors_p = [blue_p, green_p]
                        else:
                            colors_p = [blue_p, green_p, red_p]
                    elif gen_type == 'int':
                        colors_p = [blue_p, green_p, red_p]
                    
                    # define all elements in the picture
                    picture = []
                    for i in range(8): # picture has len = 8
                        picture.append([random.choice(colors_p),random.choice(shapes_p)])                    
                    
                    # uncomment if linear answer must be in the picture
                    #picture[ordnum-1] = item_p
                    
                    # count how many items are the same as the target
                    count = 0
                    for elements in range(len(picture)):
                        if picture[elements] == item_p:
                            count += 1 

                    # make sure trial is divergent on target-present trials:
                    # run through picture and count how many elements to the left of the target are the same as the target 
                    targetnum = 0
                    for elements in range(ordnum-1):
                        if picture[elements] == item_p:
                            targetnum += 1
                    
                    if trial == 'present':
                        # if there are enough items the same as the target (so the hierarchical target is present)
                        if count >= (ordnum):
                            # and if the trial is divergent
                            if targetnum != ordnum-1:
                                is_picture_divergent = True
                    elif trial == 'absent':
                        if count < (ordnum):
                            is_picture_divergent = True                            
                
                if trial == 'present':
                    # now that we know that the hierarchical target is present, find it
                    targetfound = False
                    while targetfound == False:
                        targetnum = 0 
                        # run through all elements in the picture
                        for elements in range(len(picture)):
                            # if an element is the same as the target, count it
                            if picture[elements] == item_p:
                                targetnum += 1
                                # if the ordinal of target is reached, save target position and exit the loop
                                if targetnum == ordnum:
                                    targetposition = elements+1
                                    targetfound = True
                elif trial == 'absent':
                    targetposition = 9                
                
                # save the output vector
                output_vec = [0] * 9
                output_vec[targetposition-1] = 1
                                            
                # flatten the picture, which is an array of arrays
                pict_flat = [item for sublist in picture for item in sublist]
                pict_flat = [item for sublist in pict_flat for item in sublist]
                pict_flat = np.array(pict_flat)

                # normalize the picture to make sure it has "length" 1 (length defined as Euclidean distance), as all one-hot vectors do
                # this ensures that the input to the lstm has the same "net content" across all vectors
                pict_norm = pict_flat / np.sqrt(np.sum(pict_flat**2))
                
                # add to the left of the picture 10 zeros to incorporate the length of each reduced embedding 
                pict_pad = np.pad(pict_norm,(10,0), 'constant')
                
                # add to the right of each word 64 zeros to incorporate the length of the picture
                ordinal_pad = np.pad(ordinal,(0,64), 'constant')
                color_pad = np.pad(color,(0,64), 'constant')
                shape_pad = np.pad(shape,(0,64), 'constant')
                
                # combine all three words and the picture into one flat list
                input_vec = [list(ordinal_pad),list(color_pad),list(shape_pad),list(pict_pad)]
                input_vec = [s for sublist in input_vec for s in sublist]
                
                # cut the list up into four equal-sized numpy arrays
                input_vec = np.array(input_vec)
                input_vec = np.reshape(input_vec,newshape=(4,74))
          
                # check whether the input-output combination is new:
                # compare it to all previously generated examples in input_batch and exit loop if the generated trial is new
                if runs == 0:
                    is_trial_new = True
                else:
                    ## make sure that the input picture is new/hasn't been created before
                    newness = np.zeros(runs)
                    for inputs in range(1,runs):
                        # element-wise comparison; if all are True, the two input_vecs were identical
                        if (input_vec == input_batch[inputs-1]).all() == True:
                            newness[inputs] = 1
                            break # stop looping through more runs
                            
                    if sum(newness) == 0.0: # if current input_vec is not identical to any of the previous ones, exit the loop
                        is_trial_new = True
                        
            input_batch.append(input_vec)
            output_batch.append(output_vec)

    # return input and corresponding output for training and set sets                  
    return np.array(input_batch[0:train_size]), np.array(output_batch[0:train_size]), np.array(input_batch[train_size:]), np.array(output_batch[train_size:])    
