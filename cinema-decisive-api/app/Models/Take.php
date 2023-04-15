<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Take extends Model
{
    protected $table = 'take';
    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        'id_father',
        'id_emotion_section',
        'video_path',
        'duration'
    ];
}
