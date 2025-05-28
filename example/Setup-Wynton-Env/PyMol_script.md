# PyMol script

## Set up style

Copy and paste the following script into PyMOL to setup style:

```pymol
bg_color white                           
space cmyk                               
set orthoscopic, on                      
set valence, off                         
set cartoon_side_chain_helper, on       
set cartoon_fancy_helices, on            

# Cartoon style settings with shading
set ray_trace_mode, 0
set ray_shadow, off                      
set light_count, 8                      
set ambient, 0.3                         
set reflect, 0.4                         
set direct, 0.8                         
set specular, 0                         
set ambient_occlusion_mode, 1           
set ambient_occlusion_smooth, 10          
set ambient_occlusion_scale, 15          

# DK in-house cartoon style settings
set cartoon_rect_length, 1.0             
set cartoon_oval_length, 1.0           
set stick_radius, 0.2                 
set solvent_radius, 1.6                  
set sphere_scale, 0.15                  
set dash_gap, 0.25                      
set dash_color, black                   
set mesh_width, 0.5                    

# Define custom color for the isomesh density map
set_color density_blue, [0.4, 0.6, 0.8]  

# Define custom colors for the ligands
set_color teal, [0, 0.5, 0.5]            
set_color gold, [1.0, 0.843, 0.0]        
set_color plum, [0.568, 0.239, 0.527]    
set_color cool_blue, [0.3412, 0.4587, 0.8833]
set_color cool_red, [0.8654, 0.4671, 0.3216]
set_color cool_green, [0.4569, 0.6412, 0.5725]

# OPTIONAL: Set cartoon to loop representation
# cartoon loop
```

---

## Import maps and models

Open your density map and model in PyMol:

```pymol
load /path/to/map/MAP.mtz                # Load density map
load /path/to/model/MODEL.pdb            # Load model
```

Carve your density map around the ligand of interest using the ligand name (replace ‘LIG’).  
If you do not know the ligand name, check the PDB file or click on the ligand in PyMOL:

```pymol
isomesh carved_map_name, MAP, 1.0, resn LIG, carve=1.5
```

- The numerical value after MAP is the sigma scale, adjust as needed.
- Increase/decrease carve setting if you need to show more/less map around the ligand.
- You can choose `carved_map_name` to be whatever you want.

Apply custom color to the carved isomesh:

```pymol
color density_blue, carved_map_name
```

---

## Style and color the ligand

### If the PDB is a protein-ligand complex (with a multiconformer ligand):

Right click on the ligand to select it as `(sele)`.

```pymol
select ligand, (sele)

# Select conformer A
select conformer_0, ligand and alt A

# Select conformer B
select conformer_1, ligand and alt B
```

### Alternatively, load the ligand conformers separately:

```pymol
load /path/to/model/conformer_0.pdb
load /path/to/model/conformer_1.pdb
```

---

## Clean up and color

Hide water residues:

```pymol
remove resn HOH
```

Set model carbon color to light gray:

```pymol
color hydrogen, (MODEL and elem C)
```

Color ligand atoms:

```pymol
color cool_blue, (conformer_0 and elem C)
color cool_red, (conformer_1 and elem C)
```

To use another preset color:

```pymol
color COLOR, (ligand and elem C)
```

- See: [https://pymolwiki.org/index.php/Color_Values](https://pymolwiki.org/index.php/Color_Values)

---

## Hide/show cartoon representations

To hide protein cartoon:

```pymol
hide cartoon
```

To show cartoon again:

```pymol
show cartoon
```

---

## Adjust the view and save image

- Manually orient and zoom the view.
- To save an image:

```pymol
png /path/to/save/image.png
```

---

[PyMol Color Values Reference](https://pymolwiki.org/index.php/Color_Values)
