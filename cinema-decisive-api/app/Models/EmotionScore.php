<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class EmotionScore extends Model
{
    protected $table = 'emotion_score';
    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        'id_emotion',
        'id_take',
        'score'
    ];
}
