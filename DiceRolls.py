with tf.Graph().as_default(), tf.Session() as sess:
  
    #random_uniform: Lower bound included, upper bound excluded
    #Dice rolls and sum
    first_dice_roll = tf.random_uniform([10,1], 1, 7, dtype = tf.int32) 
    second_dice_roll = tf.random_uniform([10,1], 1, 7, dtype = tf.int32)
    dice_roll_sum = first_dice_roll + second_dice_roll

    dice_roll_matrix = tf.concat(values=[first_dice_roll, second_dice_roll, dice_roll_sum], axis = 1)

    sess.run(tf.global_variables_initializer())

    #Evaluate and print dice_roll_matrix
    print(dice_roll_matrix.eval())
