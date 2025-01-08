import java.util.Scanner;
import java.util.Random;
class ROCKPAPERSCISSORS{
    public static void main(String[] args){

        Scanner sc=new Scanner(System.in);
        String play = "yes";
        //while loop to determine if we are going to play
        while(play.equals("yes"))
        {
          Random rand = new Random();
          int num;
          String userChoice="";
          String computerChoice="";

          System.out.println("Welcome to Rock, Paper, and Scissors!");
          System.out.print("Please choose R)ock, P)aper, or S)cissors.>");

          userChoice=sc.nextLine();

          //computer choice
          num=rand.nextInt(3);
          if(num==0){
            computerChoice="R";
            }
          else if(num==1){
            computerChoice="P";
            }
          else if(num==2){
            computerChoice="S";
            }
          //print computerchoice
          if(computerChoice.equals("S")){
            System.out.println("The computer choose Scissors.");
            }
          else if(computerChoice.equals("R")){
            System.out.println("The computer choose Rock.");
            }
          else if(computerChoice.equals("P")){
            System.out.println("The computer choose Paper.");
            }

         //determine the winner
          if(userChoice.equals("R")&&computerChoice.equals("S")){
            System.out.println("The user WON!");
            }
          else if(userChoice.equals("P")&&computerChoice.equals("R")){
            System.out.println("The user WON!");
            }
          else if(userChoice.equals("S")&&computerChoice.equals("P")){
            System.out.println("The user WON!");
            }
          else if(userChoice.equals("S")&&computerChoice.equals("R")){
            System.out.println("The computer WON!");
            }
          else if(userChoice.equals("R")&&computerChoice.equals("P")){
            System.out.println("The computer WON!");
            }
          else if(userChoice.equals("P")&&computerChoice.equals("S")){
            System.out.println("The computer WON!");
            }
          else if(userChoice.equals(computerChoice)){
            System.out.println("The Tie!");
            }
            System.out.print("Would you like to play again?Yes or No:");
            play = sc.nextLine();

        }
    }
}