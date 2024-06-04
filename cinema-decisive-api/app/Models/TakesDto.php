<?php

namespace App\Models;

class TakesDto{
    public $takesDto = [];

    public function getTakes(){
        return $this->takesDto;
    }

    public function addTake($emotion){
        $this->takesDto[] = $emotion;
    }
}
