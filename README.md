# GA_optimization

## Using genetic algorithms to solve optimization problems


The purpose of this exercise is to use genetic algorithms to solve an optimization problem. Suppose Hamrahe Aval (MCI) wants to build a large number of antennas to improve coverage across the city. The company has asked you to solve this problem as optimally as possible. Suppose for simplicity the whole city map is inside a rectangle with dimensions H * W. Suppose this company has k type antenna. The cost of making the 1st type antenna is equal to the roof and its coverage radius is equal to the number of this type available in the warehouse. An antenna covers areas of the area that are less than or equal to the antenna radius. Each point of the map will be covered if at least one antenna is overshadowed.


## Case 1: Solve the problem regardless of the antennas in the city

In this case, you have been told to assume that the city environment is empty of antennas. Using the genetic algorithm and receiving input information, including land dimensions and the above specifications, your task is to report the number of antennas used, the total cost, and the percentage of coverage of parts of the city.

## Case 2: Solve the problem according to the antennas in the city

In this case, you should consider the antennas in the city and repeat the required items in the first case for this case as well. 


## Description of input files

The first line of the input file named "input_section_1" specifies the length and width of the city. In the second line, the number of k antennas is specified, and in the next k line, the number, price and radius of each antenna are specified in each line.
The input file named "input_section_2" is the same as the file described above, with the difference that after the information about the company's antennas, the coordinates and radius of the antennas in the city are also given.
