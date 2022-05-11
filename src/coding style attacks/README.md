# author-style-transform
## Table of Contents

- [Introduction](#introduction)
- [Functionality](#functionality)
- [Operation-Steps](#operation-steps)
- [Operating-environment](#operating-environment)

## Introduction

Change the original author's coding style, involving 23 types of coding style attributes.

## Functionality
> * Targeted attack
> * Untargeted attack 

## Operation steps
1. **Data processing**.
>	*If it is a new dataset, execute the `find . -name '*.c' ! -type d -exec bash -c 'expand -t 4 "$0" > /tmp/e && mv /tmp/e "$0"' {} \;` Process the C / C + + data set or Java first. After processing, it's better to save it and replace it with the original data. After that, you don't need to execute the second command (modify '*. c' according to your own dataset)
2. **Enter transform directory**.
  >	* Execute the `python create_directory.py` command to create directories
  >	* Put the test set in "./program_file/test" directory
  >	* Place the target author style dataset in "./program_file/target_author_file” directory
  >	* Execute the `python get_style.py` command to generate the XML file of author style and program
  >		* output:  
  "./author_style" directory  
  "./xml_file" directory
  >	* Targeted attack
  >		* run `python targeted_attack.py` command  
  output directory: "./program_file/targeted_attack_file" directory
  >	* Untargeted attack
  >	  * run `python untargeted_attack.py --form=best`(The forms of transformation are best, random or all)command  
  output directory: "./program_file/untargeted_attack_file"

## Operating environment
> * Ubuntu environment
> * Srcml (https://www.srcml.org/)
> * Python3 environment

