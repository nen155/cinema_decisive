<?php

namespace App\Http\Controllers;

use App\Models\Movie;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class MovieController extends Controller
{
    public function __construct()
    {

    }

    public function show(){
        return Movie::all();
    }

    public function create(Request $request){
        try{

            $movie = json_decode($request->movie);

            $movieCreated = Movie::create([
                'name'=> $movie->name,
                'duration'=> $movie->duration,
                'number_scenes' => $movie->number_scenes
            ]);

            $imageObj = $this->saveImage($request, $movieCreated->id);

            if(isset($imageObj["image"])){
                $movieCreated->image_path = "movies/".$movieCreated->id."_". $imageObj["path"];
                $movieCreated->image_name = $imageObj["name"];
                $movieCreated->save();
            }

        }catch(\Exception $exception){
            error_log($exception);
        }
    }

    public function update(Request $request){
        try{
            $movieRequest = json_decode($request->movie);

            $movie = Movie::where('id',$movieRequest->id)->firstOrFail();

            $imageObj = $this->saveImage($request, $movie->id);

            $movie->name = $movieRequest->name;
            $movie->duration = $movieRequest->duration;
            $movie->number_scenes = $movieRequest->number_scenes;

            if(isset($imageObj["image"])){
                $movie->image_path = "movies/".$movieRequest->id."_". $imageObj["path"];
                $movie->image_name = $imageObj["name"];
            }
            return $movie->save();
        }catch(\Exception $exception){
            error_log($exception);
        }
    }

    public function delete($id){
        try{
            return DB::table('movie')->where('id', '=', $id)->delete();
        }catch(\Exception $exception){
            error_log($exception);
        }
    }

    private function saveImage(Request $request, $idMovie){
        $image = $request->file('image');
        if(isset($image)){
            $name = $image->getClientOriginalName();
            $image->store('public/uploads');
            $image->move(public_path('movies'), $idMovie."_". $name);
            $path = $name;
        }
        return ["image"=>$image, "path" => $path, "name" => $name];
    }
}
