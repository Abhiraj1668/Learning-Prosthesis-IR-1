#include <Arduino.h>
#include <Servo.h>

//Object of all servos created.
Servo index;    //D2
Servo middle;   //D5
Servo ring;     //D3
Servo little;   //D6
Servo thumb;    //D4
Servo hand;     //D7

//Store Serial input
int gestureIn;
int flag = 0;

//Structure to store positions of each finger which is to be moved
struct fingPos
{
  int indPos, midPos, rinPos, litPos, thuPos, hanPos;
}allPos;


//This Function uses the previous structure to pass respective position values to each motor
void gesture(){
  index.write(0);
  middle.write(0);
  ring.write(0);
  little.write(0);
  thumb.write(0);
  hand.write(120);
  delay(1000);
  index.write(allPos.indPos);
  delay(100);
  middle.write(allPos.midPos);
  delay(100);
  ring.write(allPos.rinPos);
  delay(100);
  little.write(allPos.litPos);
  delay(100);
  thumb.write(allPos.thuPos);
  delay(100);
  Serial.println("Gesture Complete!!! ");
}

//This function rotates the motor from 0-120 degrees for testing purpose
void movementTest(Servo *finger){
        for (size_t i = 0; i < 120; i++)
        {
                Serial.println(i);
                finger->write(i);
                delay(150);
        }
        finger->write(0);
}

//This function is for self testing the motors and check for the correct working of connected mechanical parts
void selftest(){
        Serial.println("Self Test - START");
        Serial.println("Testing Index Finger");
        movementTest(&index);
        Serial.println("Testing Middle Finger");
        movementTest(&middle);
        Serial.println("Testing Ring Finger");
        movementTest(&ring);
        Serial.println("Testing Little Finger");
        movementTest(&little);
        Serial.println("Testing Thumb Finger");
        movementTest(&thumb);
        Serial.println("Testing Hand Finger");
        movementTest(&hand);
        hand.write(120);
        Serial.println("Self Test - END\n");
}

void setup() {
  //Setup all fingers connected to corresponding pin number on arduino
  index.attach(2);
  middle.attach(5);
  ring.attach(3);
  little.attach(6);
  thumb.attach(4);
  hand.attach(7);
  //Serial Communication begin to communicate with PC, for debugging purpose
  Serial.begin(9600);
  //Set all fingers to open wide as mechanically possible.
  index.write(0);
  middle.write(0);
  ring.write(0);
  little.write(0);
  thumb.write(0);
  hand.write(120);
  selftest();
  Serial.println("Enter alphabet to see the corresponding gesture");
}

void loop() {
  //Print the statement only once
  if (flag == 0)
  {
    Serial.println("a:ZERO, b:ONE, c:TWO, d:THREE, e:FOUR, f:FIVE, g:SISSORS, h:PAPER, i:STONE, j:SPIDER-MAN, k:ROCK, l:THUMBS-UP, m:OK--- ");
    flag = 1;
  }
  
  //If user input available, then go execute the following statements
  if (Serial.available() > 0)
  {
    flag = 0;
    //Read gesture input from user
    gestureIn = Serial.read();
    //Set finger position of each finger according to required gesture
    switch (gestureIn)
    {
    case 'i': //Stone
    case 'a': //Zero
            allPos.indPos = 120;
            allPos.litPos = 120;
            allPos.midPos = 120;
            allPos.rinPos = 120;
            allPos.thuPos = 120;
            allPos.hanPos = 120;
            break;

    case 'b': //One
            allPos.indPos = 0;
            allPos.litPos = 120;
            allPos.midPos = 120;
            allPos.rinPos = 120;
            allPos.thuPos = 120;
            allPos.hanPos = 120;
            break;

    case 'g': //Sissors
    case 'c': //Two
            allPos.indPos = 0;
            allPos.litPos = 120;
            allPos.midPos = 0;
            allPos.rinPos = 120;
            allPos.thuPos = 120;
            allPos.hanPos = 120;
            break;

    case 'd': //Three
            allPos.indPos = 0;
            allPos.litPos = 120;
            allPos.midPos = 0;
            allPos.rinPos = 0;
            allPos.thuPos = 120;
            allPos.hanPos = 120;
            break;

    case 'e': //Four
            allPos.indPos = 0;
            allPos.litPos = 0;
            allPos.midPos = 0;
            allPos.rinPos = 0;
            allPos.thuPos = 120;
            allPos.hanPos = 120;
            break;


    case 'h': //Paper
    case 'f': //Five
            allPos.indPos = 0;
            allPos.litPos = 0;
            allPos.midPos = 0;
            allPos.rinPos = 0;
            allPos.thuPos = 0;
            allPos.hanPos = 120;
            break;

    case 'j': //Spiderman
            allPos.indPos = 0;
            allPos.litPos = 0;
            allPos.midPos = 120;
            allPos.rinPos = 120;
            allPos.thuPos = 0;
            allPos.hanPos = 120;
            break;

    case 'k': //Rock
            allPos.indPos = 0;
            allPos.litPos = 0;
            allPos.midPos = 120;
            allPos.rinPos = 120;
            allPos.thuPos = 120;
            allPos.hanPos = 120;
            break;

    case 'l': //Thumbs Up
            allPos.indPos = 120;
            allPos.litPos = 120;
            allPos.midPos = 120;
            allPos.rinPos = 120;
            allPos.thuPos = 0;
            allPos.hanPos = 120;
            break;

    case 'm': //OK
            allPos.indPos = 120;
            allPos.litPos = 0;
            allPos.midPos = 0;
            allPos.rinPos = 0;
            allPos.thuPos = 120;
            allPos.hanPos = 120;
            break;

    default:
      break;
    }
    //Write the positions to the motors
    gesture();
  }
}