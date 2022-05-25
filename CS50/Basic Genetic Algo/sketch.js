const GENE_LENGTH = 400;
const POPULATION_SIZE = 100;
const MUTATION_RATE=0.02;

var frames =0;
var population =[];//population is an array containing all the nodes 
var pool =[];
var averageFitness =0;
var generationCount =0;

function setup() {
  createCanvas(600, 550);
  for(let i =0; i < POPULATION_SIZE; i++) {
    this.population[i] = new Node(GENE_LENGTH);
  }
  target = createVector(280,5);
}

function draw() {
  //set up the background and target 
  background(52, 237, 154);
  stroke(0)
  fill(227, 227, 84)
  rect(280,5,20,20) // target
  frames++;

  if (frames % 400 === 0) {
    for(let i =0; i < POPULATION_SIZE; i++) {
      //Step 2a - Selection: Evaluate the fitness of each node 
      var temp = population[i].calcFitness();
      averageFitness = averageFitness + temp;
    }
    //Step 2b - Selection: Generate a mating pool
    matingPool();

    //Step 3: Reproduction 
    reproduce();
    frames =0;
    generationCount++;

    //count average
    averageFitness = averageFitness/POPULATION_SIZE;

  }
    
  //Step 1: Initialize the population
  for(let i =0; i < POPULATION_SIZE; i++) {
    population[i].update();
    population[i].show();
  }  
  
  text(`Generation: ${generationCount}`, 25, height - 80);
  text(`Average Fitness: ${averageFitness}`, 25, height - 50);
  text(`Frames: ${frames}`, 25, height - 20);
}

function matingPool() {
  pool =[];
  for (let i =0; i <population.length; i++) {
      var n= floor(population[i].fitness * 100); // number of times it enters the mating pool
      for (let j=0; j<n;j++){
          pool.push(population[i]); //push into pool n times based on score
      }
  }

}

function reproduce(){
  for (let i=0; i<POPULATION_SIZE;i++){
      var a = floor(random(pool.length));
      var b = floor(random(pool.length));
      var father = pool[a];
      var mother = pool[b];
      var child = father.crossover(mother);
      child.mutate(MUTATION_RATE);
      population[i]= child;
  }
}
