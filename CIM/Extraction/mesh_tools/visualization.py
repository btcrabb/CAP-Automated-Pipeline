import moviepy.editor as mpy
import numpy as np

import matplotlib
cmap = matplotlib.cm.get_cmap('gist_rainbow')
from copy import deepcopy


# Some systems have the mayavi2 module referenced by different names.
from mayavi import mlab

class Figure:

    def __init__(self, figure='Default',  fgcolor=(1, 1, 1),
                 bgcolor=(0, 0, 0), size=(400, 400)):


        self.figure = mlab.figure(figure,fgcolor = fgcolor, bgcolor=bgcolor, size=size)
        self.plots = {}
        #mlab.gcf().scene.renderer.set(use_depth_peeling=True)

    def clear(self, label=None):
        if label == None:
            labels = self.plots.keys()
        else:
            labels = [label]

        mlab.figure(self.figure.name)

        for label in labels:
            mlab_obj = self.plots.get(label)
            if mlab_obj != None:
                if mlab_obj.name == 'Surface':
                    mlab_obj.parent.parent.parent.remove()
                else:
                    mlab_obj.parent.parent.remove()
                self.plots.pop(label)

    def hide(self, label):
        if label in self.plots.keys():
            self.plots[label].visible = False

    def show(self, label):
        if label in self.plots.keys():
            self.plots[label].visible = True



    def plot_surfaces(self, label, verts, facets, scalars=None, vmax = None,
                     vmin=None, color=None, rep='surface', opacity=1.0):


        if color == None:
            color = (1, 0, 0)

        self.figure.scene.disable_render = True
        mlab_obj = self.plots.get(label)
        if not  (mlab_obj == None):
            self.clear(label)

        if scalars is None:
            self.plots[label] = mlab.triangular_mesh(verts[:, 0], verts[:, 1], verts[:, 2],
                                                     facets, color=color, opacity = opacity,
                                                     representation=rep)

        else:

            if (vmax is None):
                if (vmin is None):
                    self.plots[label] = mlab.triangular_mesh(verts[:, 0],
                                                             verts[:, 1],
                                                             verts[:, 2],
                                                             facets,
                                                             scalars=scalars,
                                                             opacity=opacity)
                else:
                    self.plots[label] = mlab.triangular_mesh(verts[:, 0],
                                                             verts[:, 1],
                                                             verts[:, 2],
                                                             facets,
                                                             scalars=scalars,
                                                             vmin = vmin,
                                                             opacity=opacity)
            else:
                if vmin is None:
                    self.plots[label] = mlab.triangular_mesh(verts[:, 0],
                                                             verts[:, 1],
                                                             verts[:, 2],
                                                             facets,
                                                             scalars=scalars,
                                                             vmax = vmax,
                                                             opacity=opacity)
                else:
                    self.plots[label] = mlab.triangular_mesh(verts[:, 0],
                                                             verts[:, 1],
                                                             verts[:, 2],
                                                             facets,
                                                             scalars=scalars,
                                                             vmin = vmin,
                                                             vmax = vmax,
                                                             opacity=opacity)
            self.plots['colorbar'] = mlab.colorbar(orientation='vertical',
                        nb_labels=3)
        self.figure.scene.disable_render = False


    def plot_lines(self, label, verts, line, color=None, size=0, opacity=1.):

        if color == None:
            color = (1, 1, 1)
        if size == None:
            size = 1

        connections = np.array(line)

        self.figure.scene.disable_render = True
        mlab_obj = self.plots.get(label)
        if not (mlab_obj == None):
            self.clear(label)
        self.plots[label] = mlab.points3d(verts[:, 0], verts[:, 1], verts[:, 2], color=color, scale_factor=0,
                                          opacity=opacity)
        self.plots[label].mlab_source.dataset.lines = connections
        mlab.pipeline.surface(self.plots[label], color=color, opacity=opacity,
                                  representation='wireframe',
                                  line_width=size,
                                  name='Connections')

        self.figure.scene.disable_render = False

    def plot_points(self, label, X, color=None, size=None, mode=None, opacity=1,
                    plot_text = False, text_to_plot = []):

        mlab.figure(self.figure.name)

        if color == None:
            color = (1, 0, 0)

        if size == None and mode == None or size == 0:
            size = 1
            mode = 'point'
        if size == None:
            size = 1
        if mode == None:
            mode = 'sphere'
        if len(text_to_plot)==0 and plot_text:
            text_to_plot = range(X.shape[0])

        if isinstance(X, list):
            X = np.array(X)

        if len(X.shape) == 1:
            X = np.array([X])

        mlab_obj = self.plots.get(label)
        if not  mlab_obj == None:
            self.clear(label)
        if isinstance(color, tuple):
            self.plots[label] = mlab.points3d(X[:, 0], X[:, 1], X[:, 2], color=color, scale_factor=size, mode=mode,
                                              opacity=opacity)
        else:
            self.plots[label] = mlab.points3d(X[:, 0], X[:, 1], X[:, 2], color, scale_factor=size,
                                              scale_mode='none',
                                              mode=mode, opacity=opacity)
        if plot_text:
            if len(text_to_plot) == len(X):
                index_label = label+'_index'
                self.plot_text(index_label, X,
                           [str(index) for index in text_to_plot], size = size)

    def plot_text(self, label, X, text, size=1, color=(1, 1, 1)):

        self.figure.scene.disable_render = True

        scale = (size, size, size)
        mlab_objs = self.plots.get(label)

        if mlab_objs != None:
            if len(mlab_objs) != len(text):
                for obj in mlab_objs:
                    obj.remove()
            self.plots.pop(label)

        mlab_objs = self.plots.get(label)
        if mlab_objs == None:
            text_objs = []
            for x, t in zip(X, text):
                text_objs.append(mlab.text3d(x[0], x[1], x[2], str(t), scale=scale, color=color))
            self.plots[label] = text_objs
        elif len(mlab_objs) == len(text):
            for i, obj in enumerate(mlab_objs):
                obj.position = X[i, :]
                obj.text = str(text[i])
                obj.scale = scale

        self.figure.scene.disable_render = False

    def plot_element_ids(self, label,node_array, elem_array , elem_ids_to_plot = [],
                         size=1, colour=(1, 1, 1)):
       if len(elem_ids_to_plot) == 0:
           elem_ids_to_plot = range(elem_array.shape[0])

       for idx, element in enumerate(elem_array):
           if idx in elem_ids_to_plot:
                position = np.mean(node_array[element[-4:]], axis=0 )
                self.plot_text('{0}{1}'.format(label, idx), [position], [idx], size=size, color=colour)

    def plot_dicoms(self, label, scan):
        # scan = self._load_dicom_attributes(dicom_files)

        mlab.figure(self.figure.name)

        mlab_objs = self.plots.get(label)
        if mlab_objs == None:
            src = mlab.pipeline.scalar_field(scan.values)
            src.origin = scan.origin
            src.spacing = scan.spacing
            plane = mlab.pipeline.image_plane_widget(src,
                                                     plane_orientation='z_axes',
                                                     slice_index=int(0.5 * scan.num_slices),
                                                     colormap='black-white')
            self.plots[label] = {}
            self.plots[label]['src'] = src
            self.plots[label]['plane'] = plane
            self.plots[label]['filepaths'] = scan.filepaths
        else:
            self.plots[label]['src'].origin = scan.origin
            self.plots[label]['src'].spacing = scan.spacing
            self.plots[label]['src'].scalar_data = scan.values
            self.plots[label]['plane'].update_pipeline()
            self.plots[label]['filepaths'] = scan.filepaths


    def plot_mesh(self, label, mesh, scalars =None, vmin = None, vmax = None,
                  face_colours=(1, 0, 0), opacity=0.5, line_colour = (1, 1, 1),
                  line_size=1., line_opacity=1., node_colour=(1,0,1),
                  node_size=0, mode = 'surface'):

        norm = matplotlib.colors.Normalize(vmin=np.min(mesh.materials),
                                           vmax=np.max(mesh.materials))
        lines = mesh.get_lines()

        verts = mesh.get_nodes()
        matlist = mesh.get_materials()
        if not( mode == 'wireframe'):
            if len(np.unique(matlist)) > 1:
                if matlist is not None:
                    for indx, mat in enumerate(np.unique(matlist)):
                        elem_list = np.where(matlist==mat)[0]
                        faces = mesh.get_surface(elem_list)
                        self.plot_surfaces('{0}_{1}_faces'.format(label,mat), verts, faces,
                                  color=cmap(norm(indx))[:3],opacity=opacity)

            else:
                elem_list = np.where(matlist == matlist[0])[0]
                faces = mesh.get_surface(elem_list)
                self.plot_surfaces('{0}_faces'.format(label), verts,
                                   faces,scalars = scalars, vmax=vmax, vmin=vmin,
                                   color=face_colours, opacity=opacity)
        self.plot_lines('{0}_lines'.format(label), verts, lines, color=line_colour,
                       size=line_size, opacity=line_opacity)
        if node_size > 0 :
            self.plot_points('{0}_nodes'.format(label), verts, color=node_colour, size=node_size)

    def make_animation(self,output_file,mesh,
                       t_node_position, duration,
                       elem_groups = None,
                       shade_elem = None,
                       t_scalars = None,
                       opacity=1, vmax=None,
                       vmin= None, view =None):

        nb_frames = len(t_node_position)
        fps = int(nb_frames/duration)

        if not (t_scalars is None):
            if vmax is None:
                vmax = np.max(t_scalars)
            if vmin is None:
                vmin = np.min(t_scalars)

        def make_frame(t):
            #mlab.clf()
            new_mesh = deepcopy(mesh)
            new_mesh.set_nodes(t_node_position[int(t*fps)])
            if t_scalars is None:
                if elem_groups is None:
                    self.plot_mesh('mesh', new_mesh, opacity=opacity)
                else:
                    cmap_mesh = matplotlib.cm.get_cmap('gist_rainbow')
                    norm_mesh = matplotlib.colors.Normalize(vmin=0,
                                                       vmax=len(
                                                           elem_groups) - 1)
                    plot_model = deepcopy(new_mesh)
                    for index_group, group in enumerate(elem_groups):
                        plot_model.set_elements(new_mesh.elements[group])

                        self.plot_mesh(str(index_group), plot_model,
                                          face_colours=cmap_mesh(
                                              norm_mesh(index_group))[:3])
            else:
                if shade_elem:
                    shaded_mesh =deepcopy(new_mesh)
                    shaded_mesh.set_elements(new_mesh.elements[shade_elem])
                    self.plot_mesh('mesh_shaded', shaded_mesh, mode ='wireframe',
                                   vmax=vmax, line_opacity=0.5)
                    elements = [x for x in new_mesh.elements
                                if x not in shaded_mesh.elements]
                    new_mesh.set_elements(elements)
                    self.plot_mesh('mesh', new_mesh,
                                   scalars=t_scalars[int(t * fps)], vmin=vmin,
                                   vmax=vmax, line_opacity=0.1, opacity = 1)
                else:


                    self.plot_mesh('mesh',new_mesh,scalars= t_scalars[int(t*fps)],vmin = vmin,
                        vmax= vmax, opacity=opacity)

            if not (view is None):
                if np.isscalar(view):
                    mlab.view(view)
                elif len(view) == 2:
                    mlab.view(view[0],view[1])
                elif len(view) == 3:
                    mlab.view(view[0], view[1], view[2])
                elif len(view)==4:
                    mlab.view(view[0], view[1], view[2],
                              roll = view[3])
            mlab.move(up=-15)

            return mlab.screenshot(antialiased=True)

        animation = mpy.VideoClip(make_frame, duration=duration)
        animation.write_gif(output_file, fps = fps)


    def close_all(self):
        mlab.close(all = True)
#    def plot_image_stack(self,image_stack):
