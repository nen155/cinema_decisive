<?php

namespace App\Models;

class TakeDto{
    public $id;
    public $id_father;
    public $video_path;
    public $image_path;
    public $id_scene;
    public $duration;
    public $emotions = [];

    public function &getEmotions(){
        return $this->emotions;
    }

    public function addEmotion($emotion){
        $this->emotions[] = $emotion;
    }

    public function __construct($id, $id_father, $video_path, $image_path, $id_scene, $duration )
    {
        $this->id = $id;
        $this->id_father = $id_father;
        $this->video_path = $video_path;
        $this->image_path = $image_path;
        $this->id_scene = $id_scene;
        $this->duration = $duration;

    }

}
