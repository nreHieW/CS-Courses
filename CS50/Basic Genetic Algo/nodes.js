//Each node represents one circle. Each node contains a DNA
const SPEED =3;

class Node {
      constructor() {
        this.dnaSequence = new DNA(GENE_LENGTH);
        this.pos = createVector(300,500); //position of a node (300,500) being thestarting spot
        this.r= 15; //node's radius 
        this.vel = p5.Vector.random2D(); // the vel vector is the vector that we will add to the position vector
        this.index = 0;
        this.fitness=0;
    }
      
      show() {
        if (this.pos.x<290 && this.pos.x>270 && this.pos.y>0 && this.pos.y<25){
          noStroke();
          fill(6, 79, 15);
          ellipse(this.pos.x, this.pos.y, this.r);
        } else {
          noStroke();
        fill(37, 168, 194);
        ellipse(this.pos.x, this.pos.y, this.r);
        }
        
      }
      
      update() {
        this.vel.add(this.dnaSequence.genes[this.index]);
        this.vel.limit(SPEED);
        this.pos.add(this.vel);
        this.index++;
      }

      calcFitness() {
        this.fitness = this.dnaSequence.calcFitness(this.pos);
        return this.fitness;
      }

      crossover(partner) {
        var child = new Node(GENE_LENGTH);
        child.dnaSequence=this.dnaSequence.crossover(partner.dnaSequence);
        return child;
        
      }

      mutate(rate){
          this.dnaSequence.mutate(rate);
      }
}