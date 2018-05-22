# coding: utf-8

# -----------------------------------------------------------------------------
# Autor: Mamaev Alexander <aamamaev.post@gmail.com>
# -----------------------------------------------------------------------------

import numpy as np
import copy
from collections import namedtuple, defaultdict
from statsmodels.stats.proportion import proportion_confint
from functools import partial
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import *
from ipywidgets import IntProgress
from IPython.display import display


class Combiner:

    def __init__(self):
        self.auto_ds = None
        self.hand_df = None

    def fit(self, X, Y):
        self.auto_ds = DataSet(X, Y)
        self.hand_df = copy.deepcopy(self.auto_ds) 
    
    def transform(self, X, hand=True):
        X_transform = X.copy()

        if hand:
            self.hand_df.transform(X, X_transform)
        else:
            self.auto_ds.transform(X, X_transform)
        
        return X_transform
    
    def vfit(self):
        root = tk.Tk()   
        # root.geometry('2000x800')
        root.title('Combiner')
        mf = MainFrame(self.hand_df)
        mf.pack(expand=YES, fill=BOTH)
        mainloop()


class Feature:

    PROC_THR = 0.5e-2  # 0.5% Категории объёмом менее PROC_THR объединяются в одну.
    INTERVAL_CNT = 20  # Количество диапазонов разбиения интервальной переменной.
    MERGE_THR = 0.5e-2  # 0.5% Группы с разницей точечных оценок долей "1" менее MERGE_THR объединяются в одну.

    def __init__(self, feature, dataset):
        self.name = feature 
        self.dataset = dataset
        self.groups = []
    
    @property
    def iv(self):
        return 0.


class DatetimeFeature(Feature):
    pass


class IdFeature(Feature):
    pass


class EmptyFeature(Feature):
    pass

    
class InputFeature(Feature):
    
    def __init__(self, feature, dataset):
        super().__init__(feature, dataset)
  
    @property
    def iv(self):
        return sum([group.iv for group in self.groups]) 
    
    @staticmethod
    def isintersect(pair):
        # Функция определят пересекаются ли доверительные интервалы на долю выбранной пары групп. 
        # Доверительные интервалы пересекаютя, если в отсортированном по возрастанию списке границ интервалов
        # левые границы следют друг за другом. 
        x1, x2, y1, y2 = pair.first.left, pair.first.right,\
                         pair.second.left, pair.second.right
        return not set(sorted([x1, x2, y1, y2])[:2]) ^ {x1, y1}
        
    def grouping(self):
        self.groups = [Group(category, label) for label, category in enumerate(self.categories)]
        
        # Пока есть группы с пересекающимися доверительными интервалами на долю "1".
        intersection = True
        while intersection:
            intersection = False
            # Сортируем группы по возрастанию доли "1".
            self.groups.sort(key=lambda x: x.event_rate) 
            # Формируем список разностей точечных оценок на долю "1" между соседними группами.
            distances = list(map(lambda x,y: x.event_rate-y.event_rate, self.groups[1:], self.groups))
            
            # Сортируем список пар групп по разности точечных оценок долей "1".
            Pair = namedtuple('Pair', ('first', 'second', 'distance'))  
            pairs = list(map(Pair, self.groups[1:], self.groups, distances)) 
            pairs.sort(key=lambda x: x.distance)      

            for pair in pairs: 
                if self.isintersect(pair) or pair.distance < self.MERGE_THR:
                    merge_group = pair.first + pair.second
                    self.groups.remove(pair.first)
                    self.groups.remove(pair.second)
                    self.groups.append(merge_group)
                    intersection = True
                    break   
                    
        for label, group in enumerate(self.groups): 
            group.label = label

    def hand_grouping(self, mapping):
        # mapping представляет собой отображение {label: categories}
        self.groups = [Group(mapping[label], label) for label in mapping]
        self.groups.sort(key=lambda x: x.event_rate) 
        for label, group in enumerate(self.groups): 
            group.label = label

    def transform(self, X, X_transform):
        for group in self.groups:
            group.transform(X, X_transform)

    def printer(self):
        for group in self.groups:
            print(group)

    def __str__(self):
        header = '____' + self.name + '____\n'
        return header + '\n'.join([cat.__str__() for cat in self.categories])


class ConFeature(InputFeature):
    pass


class CatFeature(InputFeature):

    def __init__(self, feature, dataset):
        super().__init__(feature, dataset)

        limit = self.PROC_THR * dataset.size

        unique = dataset.X[feature].value_counts(dropna=True) 
        self.rare_values = set(unique[unique < limit].index)
        self.non_rare_values = set(unique[unique >= limit].index) 

        # Создание отдельных категорий для пропущенных значений и значений которые не присутствовали в
        # fit(X), но возможно появятся в transform(X1)
        self.categories = [MissingCategory(feature, dataset),
                           UnknownCategory(self.rare_values|self.non_rare_values, feature, dataset)]
        
        if self.non_rare_values:
            self.categories.extend([CatCategory(cname, feature, dataset) for cname in self.non_rare_values])

        if self.rare_values:
            self.categories.append(RareCategory(self.rare_values, feature, dataset))    
                
        self.grouping()


class Category(object):
    
    SMALL_FLOAT = 0.0001
    
    def __init__(self, mask, dataset):
        self.dataset = dataset
        self.good_cnt = sum(mask & dataset.target.good_mask)
        self.bad_cnt = sum(mask & dataset.target.bad_mask)
        self.proc_bad = self.bad_cnt / dataset.target.bad_cnt 
        self.proc_good = self.good_cnt / dataset.target.good_cnt 
        
        if self.good_cnt + self.bad_cnt > 0: 
            self.left, self.right = proportion_confint(self.good_cnt,
                                                       self.good_cnt + self.bad_cnt,
                                                       method='wilson')
            self.event_rate = self.good_cnt / (self.good_cnt + self.bad_cnt)
        else: 
            self.left, self.right = 0., 0.
            self.event_rate = 0.

    def tree_data(self):
        return list(map(str, [self.name] +
                        [round(x*100,2) for x in [self.left, self.event_rate, self.right]] +
                        [self.good_cnt+self.bad_cnt, self.good_cnt]))
            
    def __str__(self):
         return ("  Категория: {name};\n"
                 "   Count: {count};\n"
                 "   Event Rate: {event_rate:.4};").format(name=self.name,
                                                           count=self.good_cnt + self.bad_cnt,
                                                           event_rate=self.event_rate)

    def transform(self, X):
        raise NotImplementedError("Определите метод transform в %." % self.__class__.__name__)
        
        
class CatCategory(Category):
    
    def __init__(self, value, feature, dataset):
        self.value = value
        self.name = value
        self.feature = feature
        super().__init__(dataset.X[feature] == value, dataset)
    
    def transform(self, X, X_transform, label):
        X_transform.loc[X[self.feature] == self.value, self.feature] = label 
        
        
class MissingCategory(Category):
    
    MISSING = 'MISSING'
    
    def __init__(self, feature, dataset):
        self.name = self.MISSING
        self.feature = feature
        super().__init__(dataset.X[feature].isnull(), dataset) 
        
    def transform(self, X, X_transform, label):
        X_transform[self.feature].replace(np.nan, label, inplace=True)
        
        
class UnknownCategory(Category):
    
    UNKNOWN = 'UNKNOWN'
    
    def __init__(self, values, feature, dataset):
        self.name = self.UNKNOWN
        self.values = values
        self.feature = feature
        super().__init__([False]*dataset.size, dataset) 
    
    def transform(self, X, X_transform, label): 
        unique = set(X[self.feature].value_counts(dropna=True).index)
        mask = X[self.feature].isin(unique - self.values)
        X_transform.loc[mask, self.feature] = label


class RareCategory(Category):
    
    RARE = 'RARE'
    
    def __init__(self, values, feature, dataset):
        
        self.name = self.RARE
        self.values = values
        self.feature = feature
        # mask = reduce(pd.Series.__or__, [dataset.X[feature] == cname for cname in values])
        mask = dataset.X[feature].isin(values)
        super().__init__(mask, dataset) 
        
    def transform(self, X, X_transform, label):
        mask = X[self.feature].isin(self.values)
        X_transform.loc[mask, self.feature] = label


class ConCategory(Category):
        pass


class Group:
    
    SMALL_FLOAT = 0.0001
    
    def __init__(self, categories, label):
        
        self.categories = categories if isinstance(categories, list) else [categories] 
        self.target = self.categories[0].dataset.target
        self.label = label
        self.update()   
            
    def __add__(self, group):
        return Group(self.categories + group.categories, self.label)

    def __sub__(self, group):
        try:
            for category in group.categories: self.categories.remove(category)
            return Group(self.categories, self.label)

        except Exception: 
            print('Категория {} отсутсвует в данной группе.' % category.name)
            
    def update(self):
        self.good_cnt = sum([category.good_cnt for category in self.categories])
        self.bad_cnt = sum([category.bad_cnt for category in self.categories])
        
        if self.good_cnt + self.bad_cnt > 0: 
            self.left, self.right = proportion_confint(self.good_cnt,
                                                       self.good_cnt + self.bad_cnt,
                                                       method='wilson')
            self.event_rate = self.good_cnt / (self.good_cnt + self.bad_cnt)
        else: 
            self.left, self.right = 0., 0.
            self.event_rate = 0.

        self.proc_good = round(self.good_cnt / self.target.good_cnt, 4)
        self.proc_bad = round(self.bad_cnt / self.target.bad_cnt, 4)
        
        if self.proc_bad * self.proc_good > 0:
            self.woe = round(np.log(self.proc_good / self.proc_bad), 4)
            self.iv = round(self.woe * (self.proc_good - self.proc_bad), 4)
        else: 
            self.woe = 0.  # np.nan
            self.iv = 0.  # np.nan

    def transform(self, X, X_transform):
        for category in self.categories:
            category.transform(X, X_transform, self.label)

    def tree_data(self):
             # ("title", "left", "event_rate", "right", "count", "eventCount"))
        return list(map(str, [self.label] +
                        [round(x*100,2) for x in [self.left, self.event_rate, self.right]] +
                        [self.good_cnt+self.bad_cnt, self.good_cnt]))

    def __str__(self):
        header = ("\nГруппа {label}: ({left:.4},{event_rate:.4},{right:.4});"
                 "\nWOE: {woe:.3};\nIV: {iv:.3}.\n").format(label=self.label,
                                                            left=self.left,
                                                            right=self.right,
                                                            event_rate=self.event_rate,
                                                            woe=self.woe,
                                                            iv=self.iv)
        
        return header + '\n'.join([category.__str__() for category in self.categories])  
    
    
class DataSet:

    CATEGORY_THR = 20
    MISSING_THR = 0.95
    TYPES = IDF, EMPF, DTF, CATF, CONF = 'ID EMP DT CAT CON'.split()
    CLASS = IdFeature, EmptyFeature, DatetimeFeature, CatFeature, ConFeature
    MAPPING_TC = {t: c for t, c in zip(TYPES, CLASS)}
    MAPPING_CT = {c: t for t, c in zip(TYPES, CLASS)}
    
    @staticmethod
    def log_progress(sequence):
        progress = IntProgress(min=0, max=len(sequence), value=0)
        display(progress)
        for index, record in enumerate(sequence):
                progress.value = index+1
                yield record

    def __init__(self, X, Y, size=300000):
        self.feature_names = set(X.columns)
        self.features = dict()

        self.size = size if len(X) > size else len(X)
        
        index = np.random.choice(X.index, self.size, replace=False)
        self.X = X.loc[index, :]
        self.Y = Y.loc[index]
        
        good_mask = self.Y == 1
        bad_mask = self.Y == 0
        
        good_cnt = sum(good_mask)
        bad_cnt = sum(bad_mask)
        
        Target = namedtuple('Target', ('good_mask', 'bad_mask', 'good_cnt', 'bad_cnt'))
        self.target = Target(good_mask, bad_mask, good_cnt, bad_cnt)
        
        self._typing()

    def _typing(self):
        
        for feature in self.log_progress(self.feature_names):
            # nunique = self.X.nunique(dropna=False)
            null_percent = sum(self.X[feature].isnull()) / self.size
           
            if null_percent > self.MISSING_THR:  # слишком много пропусков
                self.features[feature] = EmptyFeature(feature, self)
                
            elif str(self.X[feature].dtype) in {'datetime64', 'timedelta[ns]'}:
                self.features[feature] = DatetimeFeature(feature, self)
            else:      
                # print(feature)
                value_counts = self.X[feature].value_counts()
                
                # Id
                if len(value_counts) > self.size * 0.99:  self.features[feature] = IdFeature(feature, self)
                               
                # Категориальный
                elif sum(value_counts.iloc[:self.CATEGORY_THR]) / ((1-null_percent) * self.size) > 0.8:
                    # Если суммарный объем первых CATEGORY_THR превышает 80% от всех не null значений, то признак
                    # будет отнесен к категориальным.
                    self.features[feature] = CatFeature(feature, self)
                
                # Непрерывный, Вещественный
                elif str(self.X[feature].dtype) in {'int64','float64'}: 
                    self.features[feature] = ConFeature(feature, self)

    def change_type(self, feature, ftype):
        if ftype not in self.TYPES:
            raise ValueError("Значение параметра ftype должно принадлежать множеству %s." % self.TYPES)
            
        try:
            del self.features[feature]
            self.features[feature] = self.MAPPING_TC[ftype](feature, self)
        except KeyError:
            raise ValueError("Признак c именем '%s' не найден." % feature)
                                        
    def _grouping(self):
        # map(lambda x: x.grouping() if isinstance(x, InputFeature) else None, self.features)
        for feature in self.features.values(): 
            if isinstance(feature, InputFeature):
                feature.grouping()   
        
    def transform(self, X, X_transform):
        for feature in self.features.values(): 
            if isinstance(feature, InputFeature) and feature.name in X:
                feature.transform(X, X_transform)   
                
                
class GroupingFrame(Frame):

    def __init__(self, master=None):
        super().__init__(master)
        self.feature = None
        self._setup_widgets()
        self.tree.bind('<Button-3>', self.popup)
    
    def popup(self,event):
        self.contextMenu = Menu(self, tearoff = 0)
        labels_selected_groups = set([int(self.tree.parent(i))
                                      for i in self.tree.selection()
                                      if self.tree.parent(i) != ''])
        all_labels = set([group.label for group in self.feature.groups])

        new_label = max(all_labels) + 1
        items = sorted((all_labels - labels_selected_groups) | {new_label}) 
        for i in items:
            self.contextMenu.add_command(label=i, command=partial(self.merge, i))
        self.contextMenu.tk_popup(event.x_root, event.y_root)
        
    def merge(self, label):
        categories_ids = []
        for i in self.tree.selection():
            if self.tree.parent(i) != '':
                categories_ids.append(i)
                
        mapping = defaultdict(list) 
        for group in self.feature.groups:
            for category in group.categories:
                if str(id(category)) not in categories_ids:
                    mapping[group.label].append(category)
        
        selected_cats = [category for category in self.feature.categories
                         if str(id(category)) in categories_ids]
        mapping[label].extend(selected_cats)
        
        for key in mapping:
            if not mapping[key]: 
                del mapping[key] 
        
        self.feature.hand_grouping(mapping)
        self.build_tree(self.feature)

    def _setup_widgets(self):
        container = ttk.Frame(self)
        container.pack(fill='both', expand=True)
        
        self.header = ("Название группы", "Левая граница, %", 
                                  "Event Rate, %", "Правая граница, %", 
                                  "Кол-во наблюдений", "Кол-во событий")

        self.tree = ttk.Treeview(columns=self.header)
        self.tree.column('#0', width=50)
        self.tree.column("Название группы", width=300)
  
        vsb = ttk.Scrollbar(orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.label = ttk.Label()
        self.label.grid(column=0, row=0, sticky='nw', in_=container)

        self.tree.grid(column=0, row=1, sticky='nsew', in_=container)
        vsb.grid(column=1, row=1, sticky='ns', in_=container)
        hsb.grid(column=0, row=2, sticky='ew', in_=container)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)

    def build_tree(self, feature):
        self.feature = feature
        self.tree.delete(*self.tree.get_children())
        for col in self.header:
            self.tree.heading(col, text=col.title(),
                              command=lambda c=col: self.sort_by(self.tree, c, 0))

        string = '%s : Information Value = %.3f' %(feature.name, feature.iv)
        self.label.config(text=string)

        for group in feature.groups:
            self.tree.insert('', 'end', str(group.label), values=group.tree_data(), tag='group')
            for category in group.categories:
                self.tree.insert(str(group.label), 'end', id(category),
                                 values=category.tree_data(), tag='category')
                self.tree.see(id(category))

        self.tree.tag_configure('group',   background='#CCCCCC', font=('Courier', 10, 'bold'))
        self.tree.tag_configure('category',  font=('Courier', 10))

    def sort_by(self, tree, col, descending):
        data = [(float(tree.set(child, col)), child) if col != "Название группы"
                else (tree.set(child, col), child) for child in tree.get_children('')]

        data.sort(reverse=descending)
        for ix, item in enumerate(data):
            tree.move(item[1], '', ix)

        tree.heading(col, command=lambda col=col: self.sort_by(tree, col, int(not descending)))


class FeatureFrame(Frame):
    
    def __init__(self, features=None, master=None):
        super().__init__(master)
        self.features = features
        self._setup_widgets()
        self.build_tree(features)
        
    def _setup_widgets(self):
        container = ttk.Frame(self)
        container.pack(fill='both', expand=True)

        style = ttk.Style(self)
        style.configure('Treeview', rowheight=30)

        self.header = ("Имя", "Тип", "Information Value")
        self.tree = ttk.Treeview(columns=self.header, show='headings')
        self.tree.column("Имя", width=300)

        vsb = ttk.Scrollbar(orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(column=0, row=0, sticky='nsew', in_=container)
        vsb.grid(column=1, row=0, sticky='ns', in_=container)
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(0, weight=1)
        
    @staticmethod
    def tree_data(feature):
        return feature.name, DataSet.MAPPING_CT[type(feature)], str(feature.iv)
                
    def build_tree(self, features):
        self.features = features
        self.tree.delete(*self.tree.get_children())
        for col in self.header:
            self.tree.heading(col, text=col.title(),
                              command=lambda c=col: self.sort_by(self.tree, c, 0))
 
        for feature in features.values():
            self.tree.insert('', 'end', feature.name, values=self.tree_data(feature), tag='feature')

        self.sort_by(self.tree, "Information Value", 1)
        self.tree.tag_configure('feature',  font=('Courier', 10))

    def sort_by(self, tree, col, descending):
        data = [(float(tree.set(child, col)), child) if col == "Information Value" else (tree.set(child, col), child)            for child in tree.get_children('') ]

        data.sort(reverse=descending)
        for ix, item in enumerate(data):
            tree.move(item[1], '', ix)

        tree.heading(col, command=lambda col=col: self.sort_by(tree, col, int(not descending)))


class MainFrame(Frame):
    def __init__(self, dataset, master=None):
        super().__init__(master)
        self.dataset = dataset

        self.ff = FeatureFrame(master=self, features=dataset.features)
        self.ff.pack(side=LEFT, fill=BOTH)

        self.gf = GroupingFrame(self)
        self.gf.pack(side=RIGHT, expand=YES, fill=BOTH)

        if self.ff.tree.get_children(''):
            name = self.ff.tree.get_children('')[0]
            self.gf.build_tree(self.dataset.features[name])

        self.ff.tree.bind('<Button-1>', self.show)
    
    def show(self, e):
        if self.ff.tree.selection():
            name = self.ff.tree.selection()[0]
            self.gf.build_tree(self.dataset.features[name])

