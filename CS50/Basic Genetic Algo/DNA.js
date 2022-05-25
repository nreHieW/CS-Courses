// A DNA object which itself is an array and each element being a single vector 

class DNA {
    constructor(length) {
        this.genes = [];
        for (let i = 0; i<length; i++) {
            this.genes[i] = p5.Vector.random2D();
        }
    }

    calcFitness(pos) {
        var score = 0;
        var distance = dist(pos.x,pos.y,target.x,target.y);
        score = 1- distance/height; //nearer the distance should equal higher score
        if (score<0){
            score =0;
        }
        return score;
    }

    crossover(partner) {
        var child = new DNA(GENE_LENGTH);
        for(let i =0; i<GENE_LENGTH; i++) {
            if(GENE_LENGTH/i == 0) {
                child.genes[i]=this.genes[i];
            }
            else {
                child.genes[i]=partner.genes[i];
            }
        }
        return child;
    }
    
    mutate(rate){
        for (let i=0; i<GENE_LENGTH;i++) {
            if (random(1)<rate) {
                this.genes[i] = p5.Vector.random2D();
            }
        }
    }
}
