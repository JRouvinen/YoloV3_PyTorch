############################# Description ################################
# Author: Rouvinen Juha-Matti, Insta Advance                             #
# Date: 10/04/2023                                                       #
# Updated: 24/05/2023                                                    #
############################# License ####################################
#       Copyright [2023] [Insta Advance, Juha-Matti Rouvinen]            #
#                                                                        #
#   Licensed under the Apache License, Version 2.0 (the "License");      #
#   you may not use this file except in compliance with the License.     #
#   You may obtain a copy of the License at                              #
#                                                                        #
#       http://www.apache.org/licenses/LICENSE-2.0                       #
##########################################################################

#print progressbar
def print_progress_bar(*args): #total lines, current line
    total_lines = int(args[0])
    current_line = int(args[1])
    fill = '█'
    printEnd = "\r"
    length = 50
    percent = float(current_line/total_lines*100)
    filledLength = int(length * current_line // total_lines)
    bar = fill * filledLength + '-' * (length - filledLength)
    return bar
