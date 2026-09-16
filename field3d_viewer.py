import visualpic as vp

sim_folder_path = '../ecripac2/diags/field_diag/'
#~sim_folder_path = 'examples/laser_wakefield_3d_python/diags/diag1/'
sim_code = 'openPMD'
dc = vp.DataContainer(sim_code, sim_folder_path)

dc.load_data()
print(dc.get_list_of_fields())

rho = dc.get_field('rho')
a = dc.get_field('a')
Ex = dc.get_field('Ex')

vis = vp.VTKVisualizer()
vis.add_field(rho)

vis.show()