import numpy as np
import pdb
import random

###########################################
#### generate linear training and test data
###########################################

def gen_lin(train_size = 2):

    target = []
    input_batch = []
    output_batch = []
    
    # words in one-hot encoding
    seventh =  [1,0,0,0,0,0,0,0,0]
    second =   [0,1,0,0,0,0,0,0,0]
    third =    [0,0,1,0,0,0,0,0,0]
    fourth =   [0,0,0,1,0,0,0,0,0]
    fifth =    [0,0,0,0,1,0,0,0,0]
    sixth =    [0,0,0,0,0,1,0,0,0]
    blue =     [0,0,0,0,0,0,1,0,0]
    green =    [0,0,0,0,0,0,0,1,0]
    ball =     [0,0,0,0,0,0,0,0,1]

    # combines words into their category 
    ordinals = [second, third, fourth, fifth, sixth, seventh] # 'first' not included because all trials must be divergent
    colors = [blue, green]
    shapes = [ball]
    
    # properties of elements in picture
    blue_p   = [1,0,0]
    green_p  = [0,1,0]
    ball_p   = [0,0,1]

    # combine properties of elements in picture
    colors_p = [blue_p, green_p]
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
                color = random.choice(colors)
                shape = random.choice(shapes)

                # the target item for in the picture
                item_p = [[],[]]

                # save properties of target for creating picture
                if (color == blue) == True:
                    item_p[0] = [1,0,0]
                if (color == green) == True:
                    item_p[0] = [0,1,0]
                if (shape == ball) == True:
                    item_p[1] = [0,0,1]

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
                    
                    # define all elements in the picture
                    picture = []
                    for i in range(8): # picture has len = 8
                        picture.append([random.choice(colors_p),random.choice(shapes_p)])                    
                    
                    # make sure linear answer is or is not in the picture, depending on trial type
                    if trial == 'present':
                        picture[ordnum-1] = item_p
                    elif trial == 'absent':
                        if picture[ordnum-1] == item_p:
                            if item_p == [[1,0,0],[0,0,1]]:
                                picture[ordnum-1] = [[0,1,0],[0,0,1]]
                            elif item_p == [[0,1,0],[0,0,1]]:
                                picture[ordnum-1] = [[1,0,0],[0,0,1]]

                    # count how many items are the same as the target
                    count = 0
                    for elements in range(len(picture)):
                        if picture[elements] == item_p:
                            count += 1 
                    
                    # make sure trial is divergent on target-present trials:
                    # run through picture and count how many elements to the left of the target are the same as the target 
                    targetnum = 0
                    if trial == 'present':
                        for elements in range(ordnum-1):
                            if picture[elements] == item_p:
                                targetnum += 1
                                
                    # run through picture and count how many elements are the same as the target
                    elif trial == 'absent':
                        for elements in range(len(picture)):
                            if picture[elements] == item_p:
                                targetnum += 1

                    # check whether the generated picture meets the requirements and save the correct output
                    if trial == 'present':
                        # if there are enough items the same as the target (so the hierarchical interpretation is present)
                        if count >= (ordnum):
                            # and if the trial is divergent
                            if targetnum != ordnum-1:
                                if ordnum != 7:
                                    output_vec = ordinal[:9]
                                elif ordnum == 7:
                                    output_vec = [0,0,0,0,0,0,1,0,0]
                                is_picture_divergent = True
                    elif trial == 'absent':
                        # if the hierarchical interpretation is present
                        if count >= (ordnum):
                            output_vec = [0,0,0,0,0,0,0,0,1]
                            is_picture_divergent = True

                # flatten the picture, which is an array of arrays
                pict_flat = [item for sublist in picture for item in sublist]
                pict_flat = [item for sublist in pict_flat for item in sublist]
                pict_flat = np.array(pict_flat)

                # normalize the picture to make sure it has "length" 1 (length defined as Euclidean distance), as all one-hot vectors do
                # this ensures that the input to the lstm has the same "net content" across all vectors
                pict_norm = pict_flat / np.sqrt(np.sum(pict_flat**2))
                
                # add to the left of the picture 9 zeros to incorporate the length of each word of the phrase 
                pict_pad = np.pad(pict_norm,(9,0), 'constant')
                
                # add to the right of each word 48 zeros to incorporate the length of the picture
                ordinal_pad = np.pad(ordinal,(0,48), 'constant')
                color_pad = np.pad(color,(0,48), 'constant')
                shape_pad = np.pad(shape,(0,48), 'constant')
                
                # combine all three words and the picture into one flat list
                input_vec = [list(ordinal_pad),list(color_pad),list(shape_pad),list(pict_pad)]
                input_vec = [s for sublist in input_vec for s in sublist]
                
                # cut the list up into four equal-sized numpy arrays
                input_vec = np.array(input_vec)
                input_vec = np.reshape(input_vec,newshape=(4,57))

                # check whether the input-output combination is new:
                # compare it to all previously generated examples in input_batch and exit loop if the generated trial is new
                if runs == 0:
                    is_trial_new = True
                else:
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

#################################################
#### generate hierarchical training and test data
#################################################

def gen_hier(train_size = 2):

    target = []
    input_batch = []
    output_batch = []
    
    # words in one-hot encoding
    seventh =  [1,0,0,0,0,0,0,0,0]
    second =   [0,1,0,0,0,0,0,0,0]
    third =    [0,0,1,0,0,0,0,0,0]
    fourth =   [0,0,0,1,0,0,0,0,0]
    fifth =    [0,0,0,0,1,0,0,0,0]
    sixth =    [0,0,0,0,0,1,0,0,0]
    blue =     [0,0,0,0,0,0,1,0,0]
    green =    [0,0,0,0,0,0,0,1,0]
    ball =     [0,0,0,0,0,0,0,0,1]

    # combines words into their category 
    ordinals = [second, third, fourth, fifth, sixth, seventh] # 'first' not included because all trials must be divergent
    colors = [blue, green]
    shapes = [ball]
    
    # properties of elements in picture
    blue_p   = [1,0,0]
    green_p  = [0,1,0]
    ball_p   = [0,0,1]

    # combine properties of elements in picture
    colors_p = [blue_p, green_p]
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
                color = random.choice(colors)
                shape = random.choice(shapes)

                # the target item for in the picture
                item_p = [[],[]]

                # save properties of target for creating picture
                if (color == blue) == True:
                    item_p[0] = [1,0,0]
                if (color == green) == True:
                    item_p[0] = [0,1,0]
                if (shape == ball) == True:
                    item_p[1] = [0,0,1]

                # get what the ordinal was
                for ord in range(9):
                    if ordinal[ord] == 1:
                        ordnum = ord + 1
                        if ordnum == 1: 
                            ordnum = 7 # seventh replaces first

                ### generate a picture
                
                # run the loop to make sure the trial is divergent: 
                # the hierarchical interpretation is not the linear interpretation (if hierarchical target is present)
                is_picture_divergent = False
                while is_picture_divergent == False:
                    
                    # define all elements in the picture
                    picture = []
                    for i in range(8): # picture has len = 8
                        picture.append([random.choice(colors_p),random.choice(shapes_p)])                    
                    
                    # make sure linear answer is in the picture
                    picture[ordnum-1] = item_p

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
                
                # add to the left of the picture 9 zeros to incorporate the length of each word of the phrase 
                pict_pad = np.pad(pict_norm,(9,0), 'constant')
                
                # add to the right of each word 48 zeros to incorporate the length of the picture
                ordinal_pad = np.pad(ordinal,(0,48), 'constant')
                color_pad = np.pad(color,(0,48), 'constant')
                shape_pad = np.pad(shape,(0,48), 'constant')
                
                # combine all three words and the picture into one flat list
                input_vec = [list(ordinal_pad),list(color_pad),list(shape_pad),list(pict_pad)]
                input_vec = [s for sublist in input_vec for s in sublist]
                
                # cut the list up into four equal-sized numpy arrays
                input_vec = np.array(input_vec)
                input_vec = np.reshape(input_vec,newshape=(4,57))
          
                # check whether the input-output combination is new:
                # compare it to all previously generated examples in input_batch and exit loop if the generated trial is new
                if runs == 0:
                    is_trial_new = True
                else:
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

#############################################
### generate ambiguous training and test data
#############################################

def gen_amb(train_gen = True, train_size = 2):

    target = []
    input_batch = []
    output_batch = []
    linear_batch = []

    # words in one-hot encoding
    seventh =  [1,0,0,0,0,0,0,0,0]
    second =   [0,1,0,0,0,0,0,0,0]
    third =    [0,0,1,0,0,0,0,0,0]
    fourth =   [0,0,0,1,0,0,0,0,0]
    fifth =    [0,0,0,0,1,0,0,0,0]
    sixth =    [0,0,0,0,0,1,0,0,0]
    blue =     [0,0,0,0,0,0,1,0,0]
    green =    [0,0,0,0,0,0,0,1,0]
    ball =     [0,0,0,0,0,0,0,0,1]

    # combines words into their category 
    ordinals = [second, third, fourth, fifth, sixth, seventh] # 'first' not included because all trials must be divergent
    colors = [blue, green]
    shapes = [ball]
    
    # properties of elements in picture
    blue_p   = [1,0,0]
    green_p  = [0,1,0]
    ball_p   = [0,0,1]

    # combine properties of elements in picture
    colors_p = [blue_p, green_p]
    shapes_p = [ball_p]
    
    # define number of target-present and target-absent trials
    targetpresencetrain = ['present', 'absent'] * int((train_size)/2) 
    targetpresencetest = ['present', 'absent'] * int((100)/2)     
    random.shuffle(targetpresencetrain) 
    random.shuffle(targetpresencetest) 
    
    # training size is input to the function
    runs = train_size
    
    # the training data is different from the test data: if train_gen == True -> generate training data
    if train_gen == True:

        for runs in range(0,runs):
            
                trial = targetpresencetrain[runs]
                
                # run the loop until an input-output combination is generated that is new
                is_trial_new = False
                while is_trial_new == False:
                    
                    ### generate phrase
                
                    # pick random properties for target phrase
                    ordinal = random.choice(ordinals)
                    color = random.choice(colors)
                    shape = random.choice(shapes)

                    # the target item for in the picture
                    item_p = [[],[]]

                    # save properties of target for creating picture
                    if (color == blue) == True:
                        item_p[0] = [1,0,0]
                    if (color == green) == True:
                        item_p[0] = [0,1,0]
                    if (shape == ball) == True:
                        item_p[1] = [0,0,1]
                    
                    # get what the ordinal was
                    for ord in range(9):
                        if ordinal[ord] == 1:
                            ordnum = ord + 1
                            if ordnum == 1: 
                                ordnum = 7 # seventh replaces first
                    
                    ### generate a picture
                                                
                    picture = [None] * 8
            
                    # make sure that target-present trials are convergent (linear interpretation is hierarchical interpretation) and save output vector
                    if trial == 'present':
                        for elements in range(8):
                            if elements < ordnum-1:
                                # all items to the left of the target must be the same as the target
                                picture[elements] = item_p
                            elif elements == ordnum-1:
                                picture[elements] = item_p
                                if ordnum != 7:
                                    output_vec = ordinal[:9]
                                elif ordnum == 7:
                                    output_vec = [0,0,0,0,0,0,1,0,0]
                            elif elements > ordnum-1:
                                # all items to the right of the target are random
                                picture[elements] = [random.choice(colors_p), random.choice(shapes_p)]
                    
                    # target-absent trials should not contain linear or hierarchical target
                    elif trial == 'absent':
                        hierabsent = False
                        while hierabsent == False:
                            for elements in range(8): 
                                
                                #randomly choose properties for each element
                                picture[elements] = [random.choice(colors_p), random.choice(shapes_p)]
                                
                                # make sure the linear target is not there
                                if elements == ordnum-1:
                                    if color == [0,0,0,0,0,0,1,0,0]:
                                        picture[elements] = [[0,1,0], random.choice(shapes_p)]
                                    elif color == [0,0,0,0,0,0,0,1,0]:
                                        picture[elements] = [[1,0,0], random.choice(shapes_p)]

                            # check whether hierarchical target is absent
                            counter = 0
                            for checktargets in range(8):
                                if picture[checktargets] == item_p:
                                    counter += 1
                            if counter < ordnum:
                                output_vec = [0,0,0,0,0,0,0,0,1]
                                hierabsent = True

                    # flatten the picture, which is an array of arrays
                    pict_flat = [item for sublist in picture for item in sublist]
                    pict_flat = [item for sublist in pict_flat for item in sublist]
                    pict_flat = np.array(pict_flat)

                    # normalize the picture to make sure it has "length" 1 (length defined as Euclidean distance), as all one-hot vectors do
                    # this ensures that the input to the lstm has the same "net content" across all vectors
                    pict_norm = pict_flat / np.sqrt(np.sum(pict_flat**2))

                    # add to the left of the picture 9 zeros to incorporate the length of each word of the phrase 
                    pict_pad = np.pad(pict_norm,(9,0), 'constant')

                    # add to the right of each word 48 zeros to incorporate the length of the picture
                    ordinal_pad = np.pad(ordinal,(0,48), 'constant')
                    color_pad = np.pad(color,(0,48), 'constant')
                    shape_pad = np.pad(shape,(0,48), 'constant')

                    # combine all three words and the picture into one flat list
                    input_vec = [list(ordinal_pad),list(color_pad),list(shape_pad),list(pict_pad)]
                    input_vec = [s for sublist in input_vec for s in sublist]

                    # cut the list up into four equal-sized numpy arrays
                    input_vec = np.array(input_vec)
                    input_vec = np.reshape(input_vec,newshape=(4,57))

                    # check whether the input-output combination is new:
                    # compare it to all previously generated examples in input_batch and exit loop if the generated trial is new
                    if runs == 0:
                        is_trial_new = True
                    else:
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

        # return input and corresponding output for training set
        return np.array(input_batch[0:train_size]), np.array(output_batch[0:train_size])
    
    # if train_gen == False -> generate test data
    elif train_gen == False:
        
        runs = 100
        for runs in range(0,runs):
        
                trial = targetpresencetest[runs]
            
                # run the loop until an input-output combination is generated that is new
                is_trial_new = False
                while is_trial_new == False:
                
                    ### generate phrase
                
                    # pick random properties for target phrase
                    ordinal = random.choice(ordinals)
                    color = random.choice(colors)
                    shape = random.choice(shapes)

                    # the target item for in the picture
                    item_p = [[],[]]

                    # save properties of target for creating picture
                    if (color == blue) == True:
                        item_p[0] = [1,0,0]
                    if (color == green) == True:
                        item_p[0] = [0,1,0]
                    if (shape == ball) == True:
                        item_p[1] = [0,0,1]

                    # get what the ordinal was
                    for ord in range(9):
                        if ordinal[ord] == 1:
                            ordnum = ord + 1
                            if ordnum == 1: 
                                ordnum = 7 # seventh replaces first

                    ### generate a picture
                   
                    # run the loop to make sure the trial is divergent: 
                    # the hierarchical interpretation is not the linear interpretation
                    is_picture_divergent = False
                    while is_picture_divergent == False:                  
                        count_left = 0
                        count_total = 0
                        
                        # define all elements in the picture
                        picture = []
                        for i in range(8): # picture has len = 8
                            picture.append([random.choice(colors_p),random.choice(shapes_p)]) 
                            
                        # count how many items are the same as the target
                        for elements in range(len(picture)):
                            if picture[elements] == item_p:
                                count_total += 1                      
                        
                        if trial == 'present':
                            
                            # put the linear target in there
                            picture[ordnum-1] = item_p

                            # count how many items to the left of the target are the same as the target
                            for elements in range(ordnum):
                                if picture[elements] == item_p:
                                    count_left += 1

                            # if not all items to left of target are the same as target (so trial is not convergent)
                            if count_left != ordnum:
                                # if there are enough items the same as the target (so the hierarchical target is present)
                                if count_total >= (ordnum):
                                    # exit the loop
                                    is_picture_divergent = True
                        
                        elif trial == 'absent':
                            # if the hierarchical interpretation is absent
                            if count_total < (ordnum):
                                # if linear interpretation is absent
                                if picture[ordnum-1] != item_p: 
                                    # exit the loop
                                    is_picture_divergent = True      

                    # find and save the target position
                    if trial == 'present':
                        # run through all items until ordnum is reached
                        targetfound = False
                        while targetfound == False:
                            targetnum = 0 
                            for elements in range(len(picture)):
                                if item_p == picture[elements]:
                                    targetnum += 1
                                    if targetnum == ordnum:
                                        targetposition = elements+1
                                        targetfound = True                            
                    elif trial == 'absent':
                        targetposition = 9
                                        
                    target = [0] * 9
                    target[targetposition-1] = 1
                    output_vec = target # this contains the hierarchical answer
                    linear_vec = target # this contains the linear answer
                    
                    # if target is present, linear_vec is not the same as output_vec
                    if trial == 'present':
                        linear_vec = [0] * 9
                        linear_vec[ordnum-1] = 1

                    # flatten the picture, which is an array of arrays
                    pict_flat = [item for sublist in picture for item in sublist]
                    pict_flat = [item for sublist in pict_flat for item in sublist]
                    pict_flat = np.array(pict_flat)

                    # normalize the picture to make sure it has "length" 1 (length defined as Euclidean distance), as all one-hot vectors do
                    # this ensures that the input to the lstm has the same "net content" across all vectors
                    pict_norm = pict_flat / np.sqrt(np.sum(pict_flat**2))

                    # add to the left of the picture 9 zeros to incorporate the length of each word of the phrase 
                    pict_pad = np.pad(pict_norm,(9,0), 'constant')

                    # add to the right of each word 48 zeros to incorporate the length of the picture
                    ordinal_pad = np.pad(ordinal,(0,48), 'constant')
                    color_pad = np.pad(color,(0,48), 'constant')
                    shape_pad = np.pad(shape,(0,48), 'constant')

                    # combine all three words and the picture into one flat list
                    input_vec = [list(ordinal_pad),list(color_pad),list(shape_pad),list(pict_pad)]
                    input_vec = [s for sublist in input_vec for s in sublist]

                    # cut the list up into four equal-sized numpy arrays
                    input_vec = np.array(input_vec)
                    input_vec = np.reshape(input_vec,newshape=(4,57))

                    # check whether the input-output combination is new:
                    # compare it to all previously generated examples in input_batch and exit loop if the generated trial is new
                    if runs == 0:
                        is_trial_new = True
                    else:
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
                linear_batch.append(linear_vec)
        
        # return input and two corresponding outputs (hierarchical and linear) for test set         
        return np.array(input_batch[0:100]), np.array(output_batch[0:100]), np.array(linear_batch[0:100])

#############################################
### generate mixed training and test data
#############################################

def gen_mixed(train_gen = True, ratio_amb = 50, ratio_hier = 50):

    train_size = ratio_amb + ratio_hier
    
    target = []
    input_batch = []
    output_batch = []
    linear_batch = []

    # words in one-hot encoding
    seventh =  [1,0,0,0,0,0,0,0,0]
    second =   [0,1,0,0,0,0,0,0,0]
    third =    [0,0,1,0,0,0,0,0,0]
    fourth =   [0,0,0,1,0,0,0,0,0]
    fifth =    [0,0,0,0,1,0,0,0,0]
    sixth =    [0,0,0,0,0,1,0,0,0]
    blue =     [0,0,0,0,0,0,1,0,0]
    green =    [0,0,0,0,0,0,0,1,0]
    ball =     [0,0,0,0,0,0,0,0,1]

    # combines words into their category 
    ordinals = [second, third, fourth, fifth, sixth, seventh] # 'first' not included because all trials must be divergent
    colors = [blue, green]
    shapes = [ball]
    
    # properties of elements in picture
    blue_p   = [1,0,0]
    green_p  = [0,1,0]
    ball_p   = [0,0,1]

    # combine properties of elements in picture
    colors_p = [blue_p, green_p]
    shapes_p = [ball_p]
    
    # define number of target-present and target-absent trials
    targetpresencetrain = ['present', 'absent'] * int((train_size)/2) 
    targetpresencetest = ['present', 'absent'] * int((100)/2)     
    random.shuffle(targetpresencetrain) 
    random.shuffle(targetpresencetest) 

    # define number of training trials which are ambiguous or unambiguously hierarchical
    amborhier = []
    for amb in range(ratio_amb):
        amborhier.append('amb')
    for hier in range(ratio_hier):
        amborhier.append('hier')
    random.shuffle(amborhier)
    
    runs = train_size
    
    # the training data is different from the test data: if train_gen == True -> generate training data
    if train_gen == True:

        for runs in range(0,runs):
            
                trial = targetpresencetrain[runs]
                typeoftraintrial = amborhier[runs]

                # run the loop until an input-output combination is generated that is new
                is_trial_new = False
                while is_trial_new == False:
                    
                    ### generate phrase
                
                    # pick random properties for target phrase
                    ordinal = random.choice(ordinals)
                    color = random.choice(colors)
                    shape = random.choice(shapes)

                    # the target item for in the picture
                    item_p = [[],[]]

                    # save properties of target for creating picture
                    if (color == blue) == True:
                        item_p[0] = [1,0,0]
                    if (color == green) == True:
                        item_p[0] = [0,1,0]
                    if (shape == ball) == True:
                        item_p[1] = [0,0,1]
                    
                    # get what the ordinal was
                    for ord in range(9):
                        if ordinal[ord] == 1:
                            ordnum = ord + 1
                            if ordnum == 1: 
                                ordnum = 7 # seventh replaces first
                    
                    ### generate a picture

                    # if this is an ambiguous training trial                        
                    if typeoftraintrial == 'amb':                            
                        
                        picture = [None] * 8

                    # make sure that target-present trials are convergent (linear interpretation is hierarchical interpretation) and save output vector
                        if trial == 'present':
                            for elements in range(8):
                                if elements < ordnum-1:
                                    # all items to the left of the target must be the same as the target
                                    picture[elements] = item_p
                                elif elements == ordnum-1:
                                    picture[elements] = item_p
                                    if ordnum != 7:
                                        output_vec = ordinal[:9]
                                    elif ordnum == 7:
                                        output_vec = [0,0,0,0,0,0,1,0,0]
                                elif elements > ordnum-1:               
                                    # all items to the right of the target are random
                                    picture[elements] = [random.choice(colors_p), random.choice(shapes_p)]

                        # target-absent trials should not contain linear or hierarchical target
                        elif trial == 'absent':
                            hierabsent = False
                            while hierabsent == False:
                                for elements in range(8):

                                    #randomly choose properties for each element
                                    picture[elements] = [random.choice(colors_p), random.choice(shapes_p)]

                                    # but make sure the linear target is not there
                                    if elements == ordnum-1:
                                        if color == [0,0,0,0,0,0,1,0,0]:
                                            picture[elements] = [[0,1,0], random.choice(shapes_p)]
                                        elif color == [0,0,0,0,0,0,0,1,0]:
                                            picture[elements] = [[1,0,0], random.choice(shapes_p)]

                                # check whether hierarchical target is absent
                                counter = 0
                                for checktargets in range(8):
                                    if picture[checktargets] == item_p:
                                        counter += 1
                                if counter < ordnum:
                                    output_vec = [0,0,0,0,0,0,0,0,1]
                                    hierabsent = True
                                    
                    # if this is a hierarchical training trial                        
                    elif typeoftraintrial == 'hier':                            
                        
                        # run the loop to make sure the trial is divergent: 
                        # the hierarchical interpretation is not the linear interpretation (if hierarchical target is present)
                        is_picture_divergent = False
                        while is_picture_divergent == False:

                            # define all elements in the picture
                            picture = []
                            for i in range(8): # picture has len = 8
                                picture.append([random.choice(colors_p),random.choice(shapes_p)])                    

                            # make sure linear answer is in the picture
                            picture[ordnum-1] = item_p

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

                    # add to the left of the picture 9 zeros to incorporate the length of each word of the phrase 
                    pict_pad = np.pad(pict_norm,(9,0), 'constant')

                    # add to the right of each word 48 zeros to incorporate the length of the picture
                    ordinal_pad = np.pad(ordinal,(0,48), 'constant')
                    color_pad = np.pad(color,(0,48), 'constant')
                    shape_pad = np.pad(shape,(0,48), 'constant')

                    # combine all three words and the picture into one flat list
                    input_vec = [list(ordinal_pad),list(color_pad),list(shape_pad),list(pict_pad)]
                    input_vec = [s for sublist in input_vec for s in sublist]

                    # cut the list up into four equal-sized numpy arrays
                    input_vec = np.array(input_vec)
                    input_vec = np.reshape(input_vec,newshape=(4,57))

                    # check whether the input-output combination is new:
                    # compare it to all previously generated examples in input_batch and exit loop if the generated trial is new
                    if runs == 0:
                        is_trial_new = True
                    else:
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

        # return input and corresponding output for training set
        return np.array(input_batch[0:train_size]), np.array(output_batch[0:train_size])

    # if train_gen == False -> generate test data
    elif train_gen == False:
        
        runs = 100
        for runs in range(0,runs):
        
                trial = targetpresencetest[runs]
            
                # run the loop until an input-output combination is generated that is new
                is_trial_new = False
                while is_trial_new == False:
                
                    ### generate phrase
                
                    # pick random properties for target phrase
                    ordinal = random.choice(ordinals)
                    color = random.choice(colors)
                    shape = random.choice(shapes)

                    # the target item for in the picture
                    item_p = [[],[]]

                    # save properties of target for creating picture
                    if (color == blue) == True:
                        item_p[0] = [1,0,0]
                    if (color == green) == True:
                        item_p[0] = [0,1,0]
                    if (shape == ball) == True:
                        item_p[1] = [0,0,1]

                    # get what the ordinal was
                    for ord in range(9):
                        if ordinal[ord] == 1:
                            ordnum = ord + 1
                            if ordnum == 1: 
                                ordnum = 7 # seventh replaces first

                    ### generate a picture
                   
                    is_picture_divergent = False
                    while is_picture_divergent == False:                  
                       
                        # define all elements in the picture
                        picture = []
                        for i in range(8): # picture has len = 8
                            picture.append([random.choice(colors_p),random.choice(shapes_p)]) 
                            
                        # make sure linear answer is in the picture
                        picture[ordnum-1] = item_p

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
                        # run through all items until ordnum is reached
                        targetfound = False
                        while targetfound == False:
                            targetnum = 0 
                            for elements in range(len(picture)):
                                if item_p == picture[elements]:
                                    targetnum += 1
                                    if targetnum == ordnum:
                                        targetposition = elements+1
                                        targetfound = True                            
                    elif trial == 'absent':
                        targetposition = 9
                                        
                    target = [0] * 9
                    target[targetposition-1] = 1
                    output_vec = target # this contains the hierarchical answer
                    linear_vec = target # this contains the linear answer
                    
                    # if target is present, linear_vec is not the same as output_vec
                    if trial == 'present':
                        linear_vec = [0] * 9
                        linear_vec[ordnum-1] = 1

                    # flatten the picture, which is an array of arrays
                    pict_flat = [item for sublist in picture for item in sublist]
                    pict_flat = [item for sublist in pict_flat for item in sublist]
                    pict_flat = np.array(pict_flat)

                    # normalize the picture to make sure it has "length" 1 (length defined as Euclidean distance), as all one-hot vectors do
                    # this ensures that the input to the lstm has the same "net content" across all vectors
                    pict_norm = pict_flat / np.sqrt(np.sum(pict_flat**2))

                    # add to the left of the picture 9 zeros to incorporate the length of each word of the phrase 
                    pict_pad = np.pad(pict_norm,(9,0), 'constant')

                    # add to the right of each word 48 zeros to incorporate the length of the picture
                    ordinal_pad = np.pad(ordinal,(0,48), 'constant')
                    color_pad = np.pad(color,(0,48), 'constant')
                    shape_pad = np.pad(shape,(0,48), 'constant')

                    # combine all three words and the picture into one flat list
                    input_vec = [list(ordinal_pad),list(color_pad),list(shape_pad),list(pict_pad)]
                    input_vec = [s for sublist in input_vec for s in sublist]

                    # cut the list up into four equal-sized numpy arrays
                    input_vec = np.array(input_vec)
                    input_vec = np.reshape(input_vec,newshape=(4,57))

                    # check whether the input-output combination is new:
                    # compare it to all previously generated examples in input_batch and exit loop if the generated trial is new
                    if runs == 0:
                        is_trial_new = True
                    else:
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
                linear_batch.append(linear_vec)

        # return input and two corresponding outputs (hierarchical and linear) for test set         
        return np.array(input_batch[0:100]), np.array(output_batch[0:100]), np.array(linear_batch[0:100])
