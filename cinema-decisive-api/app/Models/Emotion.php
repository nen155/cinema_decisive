<?php

namespace App\Models;

use Illuminate\Database\Eloquent\Model;

class Emotion extends Model
{
    protected $table = 'emotion';
    /**
     * The attributes that are mass assignable.
     *
     * @var array<int, string>
     */
    protected $fillable = [
        'name',
        'description'
    ];
}
